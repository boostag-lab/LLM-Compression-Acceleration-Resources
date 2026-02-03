import torch.nn as nn
import torch
import torch.nn.functional as F
import gc
import time
from utils import timeit
from tqdm import tqdm, trange
import numpy as np
import math
import copy
import random

from .d2prune_utils import D2SparseGPT, D2Wanda, D2ADMM
from .pruner_zero import PrunerZero
from .sparsegpt import SparseGPT
from .wanda import Wanda
from .admm_grad import AdmmGrad

from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class D2Prune_LLAMA:
    '''
    D2Prune:
    1. using 1st-order activation derivatives and 2nd-order weights derivatives for pruning metric
    2. attention awareness: q/k/v weights hybrid update (D2SparseGPT) or no-update (D2Wanda)
    '''

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.device  # 'cpu' or 'cuda:0
        self.sparsity_ratio = args.sparsity_ratio
        self.nsamples = args.nsamples
        self.target_layer_names = args.target_layer_names  # []
        self.d2_sparsegpt = args.d2_sparsegpt
        self.d2_wanda = args.d2_wanda
        self.d2_admm = args.d2_admm
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = self.args.logger

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.layers
        
    def init_tensorboard(self):
    # --- TensorBoard: 可通过 args 开关/自定义日志目录 ---
        self.tb_enabled = getattr(self.args, "tb_enabled", True)
        self.tb_logdir  = getattr(self.args, "tb_logdir", f"{self.args.output_dir}/runs")
        self.tb_runname = getattr(self.args, "tb_runname", f"{self.args.exp_name}")
        self.tb_writer  = getattr(self.args, "tb_writer", None)
        if self.tb_enabled and (self.tb_writer is None):
            # 每个进程/实验独立 run name（可按需改成带 model 名）
            import datetime, os
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.tb_logdir, f"{self.tb_runname}-{ts}")
            self.tb_writer = SummaryWriter(log_dir=run_dir)

        # 便捷 no-op 包装，避免到处判空
        def _tb_add(fn_name, *a, **kw):
            w = self.tb_writer
            if w is None: 
                return
            getattr(w, fn_name)(*a, **kw)
        self._tb_add = _tb_add

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params

    @staticmethod
    def check_outlier_mean(mask, threshold):
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    @staticmethod
    def check_outlier_quantile(mask, quantile=0.99):
        W = mask.flatten()
        # 计算指定分位数
        q_value = torch.quantile(W, quantile)
        # 统计大于该分位数的元素
        count = (W > q_value).sum().item()
        total_params = W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    @staticmethod
    def check_outlier_normal(mask, k=3.0):
        W = mask.flatten()
        mean = torch.mean(W)
        std = torch.std(W)
        threshold = mean + k * std
        count = (W > threshold).sum().item()
        total_params = W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    
    
    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, subset, gpts_layers, wrapped_layers, dsm='owl', granularity='per-block'):
        """
        Sparsity compensation
        Compensate for over-pruning caused by uniform sparsity due to different layer sensitivities, and balance sparsity.
        :param dsm:dynamic sparsity method-->global static adjustments
        :return:subset each layer sparsity
        """

        gpts = None
        if dsm == "owl":
            # self.layer_wmetric = []
            if granularity == 'per-block':
                self.layer_outlier_ratios = []
                self.block_sizes = []
                for name in subset:
                    # W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                                gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        # (lambda_1 ywx)^(1/2)
                        # W_metric += (self.r1 ** (1/2)) * (self.y_scaler_col.reshape((-1, 1)) ** (1 / 2)) * (torch.abs(self.layer.weight.data) ** (1 / 2)) * self.delta_x_scaler_row.reshape((1, -1)) ** (1/2)
                        # # (lambda_2 w^2x^2)^1/2=sqrt(lambda_2) wx
                        # W_metric += -(self.r2 ** (1/2)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))  # 768,128

                        # # new-->not scaling to sqrt: correct
                        ## ywx
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        ## w^2x^2
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                    gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))  # 768,128
                    block_outlier_ratio = self.check_outlier_mean(torch.flatten(W_metric.cpu()),
                                                                  self.args.Hyper_m)  # why each has the same Hyper_m
                    self.layer_outlier_ratios.append(block_outlier_ratio)
                    self.block_sizes.append(subset[name].weight.numel())  
                total_params = sum(self.block_sizes)
                block_weights = np.array(self.block_sizes) / total_params
                self.all_blocks_ratio = np.array(self.layer_outlier_ratios)
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / (self.all_blocks_ratio.max() - self.all_blocks_ratio.min())
                # [target_sparsity - lambda, target_sparsity + lambda]
                target_sparsity = self.args.sparsity_ratio
                delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
                self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)  

                # 3. WEIGHTED CALIBRATION: ensure layer sparsity strictly equals target
                current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
                scale = target_sparsity / current_weighted_sparsity
                self.all_blocks_ratio = 1-np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

                self.logger.info(f"Block sparsity: {1-self.all_blocks_ratio}, "
                                 f"Block outlier ratio: {self.all_blocks_ratio}, "
                                 f"Target sparsity: {target_sparsity:.4f}, "
                                 f"Weighted sparsity: {np.sum((1-self.all_blocks_ratio) * block_weights):.4f}, ")
                self.logger.info("before layer sparsity compensation", self.layer_outlier_ratios)
                return self.all_blocks_ratio
            elif granularity == 'per-layer':
                self.layer_wmetric = []
                for name in subset:
                    # W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                            gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        # (lambda_1 ywx)^(1/2)
                        # W_metric += (self.r1 ** (1/2)) * (self.y_scaler_col.reshape((-1, 1)) ** (1 / 2)) * (torch.abs(self.layer.weight.data) ** (1 / 2)) * self.delta_x_scaler_row.reshape((1, -1)) ** (1/2)
                        # # (lambda_2 w^2x^2)^1/2=sqrt(lambda_2) wx
                        # W_metric += -(self.r2 ** (1/2)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))  # 768,128

                        # # new-->not scaling to sqrt: correct
                        ## ywx
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                            gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        ## w^2x^2
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))  # 768,128
                    self.layer_wmetric.append(torch.flatten(W_metric.cpu()))
                self.layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in self.layer_wmetric])
                self.out_ratio_layer = self.check_outlier_mean(self.layer_wmetric, self.args.Hyper_m)
                return self.out_ratio_layer

    # === PATCH 1: helper functions (add inside class D2Prune_OPT) ===
    def _project_to_constraint(self, r, N, p_global):
        # r, N: 1D tensors on same device
        alpha = (N @ r - p_global * N.sum()) / (N @ N + 1e-12)
        return r - alpha * N
    
    def _sample_z_tangent(self, N):
        # gaussian then project to tangent space of N^T r = const
        self._seed_all(self.args.seed)
        n = N.numel()
        z = torch.randn(n, device=N.device, dtype=N.dtype)
        # z = torch.normal(mean=0, std=1, size=N.size(), device=N.device, dtype=N.dtype)
        # # project to {v | N^T v = 0}
        z = z - ( (N @ z) / (N @ N + 1e-12) ) * N
        # # normalize to unit l2
        z = z / (z.norm() + 1e-12)
        return z

    @staticmethod
    def _seed_all(seed: int):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)    
            
    def _compute_local_bounds(self, global_sparsity, layer_idx, num_layers, r_min, r_max):
        gs = float(global_sparsity)

        # 原先的带宽
        base_band = max(0.10, 0.30 * min(gs, 1.0 - gs))  # 稍放宽, 0.1-0.15
        if num_layers <= 1:
            scale = 1.0
        else:
            pos = layer_idx / max(1, num_layers - 1)
            scale = 1.0 if pos < 1/3 else (0.9 if pos < 2/3 else 0.8)

        band = base_band * scale
        r_lo = max(r_min, gs - band)
        r_hi = min(r_max, gs + band)

        # —— 最小可探索带宽兜底（关键） ——
        MIN_BAND = 0.05  # 至少允许 ±5%
        if r_hi - r_lo < MIN_BAND:
            mid = min(r_max, max(r_min, gs))
            r_lo = max(r_min, mid - MIN_BAND/2)
            r_hi = min(r_max, mid + MIN_BAND/2)
            if r_hi <= r_lo:  # 再次兜底
                r_lo = max(r_min, gs - 0.03)
                r_hi = min(r_max, gs + 0.03)

        return float(r_lo), float(r_hi)
    def mezo_dsm(self, global_sparsity, inps, outs, attention_mask, position_ids, position_embeddings, layer,
                subset, zo_eps=2e-2, epochs=1, lr=2e-1, n_spsa=1,
                r_min=0.10, r_max=0.98):
        """
        MeZO-DSM on per-Block sparsity vector r  (q/k/v/o, gate/up/down...)
        Key changes:
        - tangent-space noise under N-weighted global constraint
        - CRN: same seed & identical ADMM settings for L+/L-
        - consistent denominator (2*eps_r)
        - Euclidean projection to constraint + box clamp
        - simple precondition by 1/(N_i / meanN)
        """
            
        names = list(subset.keys())
        n = len(names)
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        dtype = torch.float32
        # —— 计算“本层”的自适应上下界（围绕 global_sparsity）——
        # ---- 在 mezo_dsm 的变量初始化后加上（如果没有的话）----
        if not hasattr(self, "_r_momentum"):
            self._r_momentum = None
        try:
            layer_idx = int(str(self.index_layer).split('_')[-1])
            num_layers = len(self.layers)
        except Exception:
            layer_idx, num_layers = 0, 1

        r_min_local, r_max_local = self._compute_local_bounds(
            global_sparsity=self.args.sparsity_ratio if isinstance(global_sparsity, list) else global_sparsity,
            layer_idx=layer_idx,
            num_layers=num_layers,
            r_min=r_min, r_max=r_max
        )
        # parameter counts per sub-layer
        with torch.no_grad():
            N = torch.tensor([subset[name].weight.numel() for name in names],
                            device=device, dtype=dtype)
            if isinstance(global_sparsity, list): # block sparsity
                r = torch.tensor(global_sparsity, device=device, dtype=dtype)
            else:
                r = torch.full((n,), float(global_sparsity), device=device, dtype=dtype)
            if isinstance(global_sparsity, list): # block sparsity
                r = self._project_to_constraint(r, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
            else:
                r = self._project_to_constraint(r, N, global_sparsity).clamp_(r_min_local, r_max_local)
            # print(f"layer {layer_idx} initial sparsity: {r.detach().cpu().tolist()}")
            self.logger.info(f"layer {layer_idx} initial sparsity: {r.detach().cpu().tolist()} "
                             f"r(min/mean/max)=({r.min().item():.3f}/{r.mean().item():.3f}/{r.max().item():.3f}) "
                             f"weighted={float((r*N).sum()/N.sum()):.4f}")
        # === 在 mezo_dsm() 里，计算出 layer_idx 之后，定义一个 tensorboard 基础 tag ===
        base_tag = f"mezo/layer_{layer_idx}"

        # （可选）记录初始化 r 与 dense baseline
        try:
            # 轻量：仅统计/直方图，不写大张量
            self._tb_add("add_scalar", f"{base_tag}/init_weighted_r", float((r*N).sum()/N.sum()), 0)
            self._tb_add("add_scalar", f"{base_tag}/r_min", float(r.min().item()), 0)
            self._tb_add("add_scalar", f"{base_tag}/r_max", float(r.max().item()), 0)
            self._tb_add("add_histogram", f"{base_tag}/r_hist", r.detach().cpu().numpy(), 0)
        except Exception:
            pass

        try:
            init_loss = eval_avg_loss(r, max_eval_samples=min(16, inps.shape[0]), seed_for_all=self.args.seed)
            self._tb_add("add_scalar", f"{base_tag}/epoch_loss", float(init_loss), 0)
        except Exception:
            init_loss = None
            

        # precompute dense outputs for CRN (already done outside, keep your logic)
        dense_Y = torch.zeros_like(outs)
        for j in range(inps.shape[0]):
            with torch.no_grad():
                # dense_Y[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

                dense_Y[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

        # ---- inner: loss after pruning this block with sparsity r ----
        def prune_and_loss(current_r, X, Y, seed_for_all):
            # CRN: same seeds & deterministic flags
            self._seed_all(seed_for_all)
            # copy the block & collect stats
            layer_copy = copy.deepcopy(layer)
            # subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, GPT)
            # subset_copy, gpts_copy, wrapped_layers_copy = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids)
            if self.args.use_wanda_forward:
                subset_copy, gpts_copy, wrapped_layers_copy = self.forward_layer_wrapper1(layer, inps, outs, attention_mask, position_ids, position_embeddings)
            else:
                subset_copy, gpts_copy, wrapped_layers_copy = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings)
            
            # 目标全局稀疏度（标量），兼容 list 情况
            if isinstance(global_sparsity, list):
                gvec = torch.tensor(global_sparsity, device=device, dtype=dtype)
                p_target = (gvec * N).sum() / (N.sum() + 1e-12)
            else:
                p_target = torch.tensor(float(global_sparsity), device=device, dtype=dtype)

            # apply pruning per sub-layer
            for i, name in enumerate(subset_copy):
                sp = float(current_r[i].item())
                if not math.isfinite(sp):
                    sp = float(global_sparsity)   # 或者 (r_min_local + r_max_local)/2
                sp = float(min(r_max_local, max(r_min_local, sp)))
                try:
                    if self.args.use_wanda_forward:
                        wrapped_layers_copy[name].fasterprune(sp, self.prune_n, self.prune_m)
                        wrapped_layers_copy[name].free()
                    else:
                        if name not in self.target_layer_names:   # 更新权重的分支
                            gpts_copy[name].fasterprune(sp, self.prune_n, self.prune_m)
                            gpts_copy[name].free()
                        else:                                     # 包装层分支
                            wrapped_layers_copy[name].fasterprune(sp, self.prune_n, self.prune_m)
                            wrapped_layers_copy[name].free()
                except Exception:
                    # 一旦剪枝内部抛错（含 NaN/Inf），直接返回极大损失，避免污染本轮估计
                    layer_copy.cpu(); del layer_copy, subset_copy, gpts_copy, wrapped_layers_copy
                    torch.cuda.empty_cache()
                    return torch.tensor(1e6, device=X.device, dtype=torch.float32), torch.tensor(1e6, device=X.device, dtype=torch.float32), torch.tensor(1e6, device=X.device, dtype=torch.float32)

                
            recon_loss = 0
            sparsity_reg_loss = 0

            with torch.no_grad():
                # sparse_Y = layer_copy(X, attention_mask=attention_mask)[0]
                sparse_Y = layer_copy(X, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # 若输出存在非有限值，返回极大损失
                if not torch.isfinite(sparse_Y).all():
                    loss = torch.tensor(1e6, device=X.device, dtype=torch.float32)
                else:
                    den = float(Y.var().item())
                    if not math.isfinite(den) or den < 1e-8:
                        den = 1e-6
                    loss = torch.nn.functional.mse_loss(sparse_Y, Y) / (den+1e-6)
                # 稀疏度正则化
                if getattr(self.args, "sparse_reg", True):
                    recon_loss += loss
                    # ========== 新增：L2 正则 ==========

                    # 全局加权稀疏度与目标的偏差
                    p_now = (current_r * N).sum() / (N.sum() + 1e-12)
                    global_reg = (p_now - p_target.to(p_now.dtype))**2

                    lam_g = float(getattr(self.args, "lambda_global", 0.0))
                    sparsity_reg_loss += lam_g * global_reg

                    loss = recon_loss + sparsity_reg_loss
                    # self.sparsity_reg_loss = lam_g * global_reg
                    
            # 清理
            layer_copy.cpu(); del layer_copy, subset_copy, gpts_copy, wrapped_layers_copy
            torch.cuda.empty_cache()
            return loss, recon_loss, sparsity_reg_loss
        # ====== 评估函数：给定 r，返回在一小批输入上的平均层内 MSE ======
        def eval_avg_loss(current_r, max_eval_samples= min(16, inps.shape[0]), seed_for_all=self.args.seed):
            self._seed_all(seed_for_all)
            # 子采样以加速（也可用全体 inps）
            idxs = list(range(inps.shape[0]))[:max_eval_samples]
            losses = []
            for j_ in idxs:
                X_j = inps[j_].unsqueeze(0); Y_j = dense_Y[j_].unsqueeze(0)
                # L = prune_and_loss(current_r, X_j, Y_j, seed_for_all)
                L,_,_ = prune_and_loss(current_r, X_j, Y_j, seed_for_all)
                losses.append(float(L.item()))
            return float(np.mean(losses))
        
        # loss_best = eval_avg_loss(r, max_eval_samples=min(16, inps.shape[0]), seed_for_all=self.args.seed)
        r_best = r.clone()
        no_improve = 0
        patience = 4            # 可调：2~4
        steps = 0
        prev_epoch_loss = 1e3
        for epoch in range(epochs):
            total_loss = 0.0
            total_r_loss = 0.0
            total_s_loss = 0.0
            # === “学不动”时的临时加速器 ===
            lr_epoch, zo_eps_epoch, n_spsa_epoch = lr, zo_eps, n_spsa
            if prev_epoch_loss is not None:
                improve = prev_epoch_loss - 0.0  # 用上一轮均值初始化
            # 真正的 improve 下面算；这里先占位
            # ======= Batch version of SPSA gradient estimate =======
            batch_size = self.args.batch_size  # 可以设为 self.args.batch_mezo
            num_batches = math.ceil(inps.shape[0] / batch_size)

            for batch_idx in range(num_batches):
                batch_slice = slice(batch_idx * batch_size, min((batch_idx+1)*batch_size, inps.shape[0]))
                X_batch = inps[batch_slice]
                Y_batch = dense_Y[batch_slice]

                g_hat = torch.zeros_like(r)
                loss_plus_acc, loss_minus_acc = 0.0, 0.0
                r_loss_plus_acc, s_loss_plus_acc = 0.0, 0.0

                for _ in range(n_spsa_epoch):
                    z = self._sample_z_tangent(N)
                    eps_r = float(zo_eps_epoch)

                    if isinstance(global_sparsity, list):
                        r_plus  = self._project_to_constraint(r + eps_r*z, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                        r_minus = self._project_to_constraint(r - eps_r*z, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                    else:
                        r_plus  = self._project_to_constraint(r + eps_r*z, N, global_sparsity).clamp_(r_min_local, r_max_local)
                        r_minus = self._project_to_constraint(r - eps_r*z, N, global_sparsity).clamp_(r_min_local, r_max_local)

                    # L_plus_mean = prune_and_loss(r_plus,  X_batch, Y_batch, self.args.seed)
                    # L_minus_mean = prune_and_loss(r_minus, X_batch, Y_batch, self.args.seed)
                    L_plus_mean, r_plus_mean, s_plus_mean = prune_and_loss(r_plus,  X_batch, Y_batch, self.args.seed)
                    L_minus_mean, _, _ = prune_and_loss(r_minus, X_batch, Y_batch, self.args.seed)
                    
                    # L_plus_mean = prune_and_loss(r + eps_r*z,  X_batch, Y_batch, self.args.seed)
                    # L_minus_mean = prune_and_loss(r - eps_r*z, X_batch, Y_batch, self.args.seed)
                    


                    coeff = (L_plus_mean - L_minus_mean) / (2.0 * eps_r + 1e-12)
                    g_hat += coeff * z
                    
                    loss_plus_acc += L_plus_mean
                    loss_minus_acc += L_minus_mean
                    
                    r_loss_plus_acc += r_plus_mean
                    s_loss_plus_acc += s_plus_mean
                    
                    


                # g_hat /= float(n_spsa)
                g_hat /= float(n_spsa_epoch)
                total_loss += loss_plus_acc / max(1, n_spsa_epoch)
                
                total_r_loss += r_loss_plus_acc / max(1, n_spsa_epoch)
                total_s_loss += s_loss_plus_acc / max(1, n_spsa_epoch)
                
                # total_loss += loss_plus_acc / max(1, _n)
                # ---- 动量更新 + 投影 ----
                if self._r_momentum is None or self._r_momentum.shape != g_hat.shape:
                    self._r_momentum = torch.zeros_like(g_hat)
                beta = 0.9
                self._r_momentum = beta * self._r_momentum + (1 - beta) * (g_hat / ((N / N.mean()).clamp(min=1e-6)))
                # r = r - lr_epoch * self._r_momentum
                r = r - lr_epoch * g_hat
                    
                if isinstance(global_sparsity, list): # block sparsity
                    r = self._project_to_constraint(r, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                    r = torch.nan_to_num(r, nan=self.args.sparsity_ratio,
                                        posinf=r_max_local, neginf=r_min_local)
                else:
                    r = self._project_to_constraint(r, N, global_sparsity).clamp_(r_min_local, r_max_local)
                    r = torch.nan_to_num(r, nan=float(global_sparsity),
                                        posinf=r_max_local, neginf=r_min_local)
                steps += 1
                
            # ====== 统计改进，若几乎不降就“踹一脚” ======
            epoch_loss_mean = total_loss / num_batches

            epoch_r_loss_mean = total_r_loss / num_batches
            epoch_s_loss_mean = total_s_loss / num_batches
            
            if prev_epoch_loss is not None:
                improve = prev_epoch_loss - epoch_loss_mean
                if improve < 1e-5:
                    lr_epoch = lr * 1.5
                    zo_eps_epoch = zo_eps * 1.5
                    n_spsa_epoch = max(1, n_spsa * 2)
                else:
                    lr_epoch, zo_eps_epoch, n_spsa_epoch = lr, zo_eps, n_spsa
                if improve > 1e-5:
                    self.logger.info(f"current r is positive adjustment, please update")
                    prev_epoch_loss = epoch_loss_mean
                    r_best = r
                else:
                    self.logger.info(f"current r is negative adjustment, don't update")
                    no_improve += 1
                    if no_improve >= patience:
                        self.logger.info(f"[MEZO] early stop at epoch {epoch}, best_loss={prev_epoch_loss:.6f}")
                        break
            try:
                self._tb_add("add_scalar", f"{base_tag}/epoch_loss", float(epoch_loss_mean), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/loss_best", float(prev_epoch_loss), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/weighted_r", float((r*N).sum()/N.sum()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_min", float(r.min().item()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_max", float(r.max().item()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_mean", float(r.mean().item()), epoch+1)
                if self.args.sparse_reg:
                    self._tb_add("add_scalar", f"{base_tag}/epoch_reg_loss", float(epoch_r_loss_mean), epoch+1)
                    self._tb_add("add_scalar", f"{base_tag}/epoch_sparse_loss", float(epoch_s_loss_mean), epoch+1)
                # 直方图别太频繁；每若干 epoch 或最后一轮再记也行
                if (epoch % 5 == 0) or (epoch+1 == epochs):
                    self._tb_add("add_histogram", f"{base_tag}/r_hist", r.detach().cpu().numpy(), epoch+1)
            except Exception:
                pass

            self.logger.info(f"[MEZO] epoch={epoch} loss={epoch_loss_mean:.6f} "
                            f"r(min/mean/max)=({r.min().item():.3f}/{r.mean().item():.3f}/{r.max().item():.3f}) "
                            f"weighted={float((r*N).sum()/N.sum()):.4f}")

            # 手动轻微衰减学习率（可选）
            # lr *= 0.95

        if isinstance(global_sparsity, list): # block sparsity
            r = self._project_to_constraint(r_best, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)   
        else:
            r = self._project_to_constraint(r_best, N, global_sparsity).clamp_(r_min_local, r_max_local)
        if isinstance(global_sparsity, list): # block sparsity
            r = torch.nan_to_num(r, nan=float(self.args.sparsity_ratio),
                                posinf=r_max_local, neginf=r_min_local)
        else:
            r = torch.nan_to_num(r, nan=float(global_sparsity),
                                posinf=r_max_local, neginf=r_min_local)
        return r.detach().cpu()


    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        device = self.model.model.embed_tokens.weight.device  
        if device.type == 'cpu':
            device = self.device
            self.model.model.embed_tokens.to(device)
        else:
            device = device.index
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype  # torch.float16
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                                device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                cache["position_embeddings"] = kwargs['position_embeddings']
                raise ValueError
        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:  #
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))  # batch[0]-->[1,2048]
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache["position_embeddings"]
        self.model.config.use_cache = self.use_cache  # True
        if self.args.free:
            self.model.model.embed_tokens.to("cpu")
        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids, position_embeddings

    def forward_layer_wrapper1(self, layer, inps, outs, attention_mask, position_ids, position_embeddings):  # no position_ids for opt
        subset = self.find_layers(layer)
        gpts = {}
        wrapped_layers = {}
        # if self.args.use_wanda_forward and (self.args.dsm =='mezo' or self.args.dsm =='owl-mezo'): #save forward mezo loss computation time
        #     self.target_layer_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj','self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        # print(subset)
        target_layer_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj','self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        for name in subset:
            if name not in target_layer_names:
                if self.args.d2_sparsegpt:
                    gpts[name] = D2SparseGPT(self.args, subset[name])
                elif self.args.d2_admm:
                    gpts[name] = D2ADMM(self.args, subset[name])
                else:
                    gpts[name] = SparseGPT(self.args, subset[name])
            else:
                if self.args.d2_wanda:
                    wrapped_layers[name] = D2Wanda(self.args, subset[name])
                else:
                    wrapped_layers[name] = Wanda(self.args, subset[name])

        def add_batch_sparsegpt(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        def add_batch_wrapped_gpt(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles_sparsegpt = []
        handles_wrapped_gpt = []
        for name in subset:
            if name not in target_layer_names:
                handles_sparsegpt.append(subset[name].register_forward_hook(add_batch_sparsegpt(name)))
            else:
                handles_wrapped_gpt.append(subset[name].register_forward_hook(add_batch_wrapped_gpt(name)))
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,768]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]  # [1,2048,768)
                
        for h in handles_sparsegpt:
            h.remove()
        for h in handles_wrapped_gpt:
            h.remove()
        return subset, gpts, wrapped_layers

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids, position_embeddings):  # no position_ids for opt
        subset = self.find_layers(layer)
        gpts = {}
        wrapped_layers = {}
        # if self.args.use_wanda_forward and (self.args.dsm =='mezo' or self.args.dsm =='owl-mezo'): #save forward mezo loss computation time
        #     self.target_layer_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj','self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        for name in subset:
            if name not in self.target_layer_names:
                if self.args.d2_sparsegpt:
                    gpts[name] = D2SparseGPT(self.args, subset[name])
                elif self.args.d2_admm:
                    gpts[name] = D2ADMM(self.args, subset[name])
                else:
                    gpts[name] = SparseGPT(self.args, subset[name])
            else:
                if self.args.d2_wanda:
                    wrapped_layers[name] = D2Wanda(self.args, subset[name])
                else:
                    wrapped_layers[name] = Wanda(self.args, subset[name])

        def add_batch_sparsegpt(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        def add_batch_wrapped_gpt(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles_sparsegpt = []
        handles_wrapped_gpt = []
        for name in subset:
            if name not in self.target_layer_names:
                handles_sparsegpt.append(subset[name].register_forward_hook(add_batch_sparsegpt(name)))
            else:
                handles_wrapped_gpt.append(subset[name].register_forward_hook(add_batch_wrapped_gpt(name)))
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,768]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]  # [1,2048,768)
                
        for h in handles_sparsegpt:
            h.remove()
        for h in handles_wrapped_gpt:
            h.remove()
        return subset, gpts, wrapped_layers

    @timeit
    def prune_layer_weight(self, subset, gpts, wrapped_layers):
        for i, name in enumerate(subset):
            if self.args.dsm == "owl":
                if self.args.granularity == 'per-block':
                    self.sparsity_ratio = 1 - self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    # self.sparsity_ratio = 1-self.all_blocks_ratio[i]
                    self.logger.info(f"block sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
                elif self.args.granularity == 'per-layer':
                    self.sparsity_ratio = 1-self.all_layers_ratio[int(self.index_layer.split('_')[-1])]
                    self.logger.info(f"layer sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
            elif self.args.dsm == 'mezo' or self.args.dsm == 'owl-mezo':
                self.sparsity_ratio = self.mezo_layer_r[int(self.index_layer.split('_')[-1])][i]
                
            if name not in self.target_layer_names: # update wights
                if self.d2_sparsegpt:
                    self.logger.info(f"pruning {name} by D2-SparseGPT: r1={self.args.r1}, r2={self.args.r2}")
                elif self.d2_admm:
                    self.logger.info(f"pruning {name} by D2_Admm")
                else:
                    self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()
            else:
                if self.d2_wanda:
                    self.logger.info(f"pruning {name} by D2-Wanda: r1={self.args.r1}, r2={self.args.r2}")
                else:
                    self.logger.info(f"pruning {name} by Wanda")
                wrapped_layers[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                wrapped_layers[name].free()
            torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []
        self.mezo_layer_r= []
        for i in tqdm(range(len(self.layers)), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)

            elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                dev = self.device
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)

            start = time.time()
            # 1. forward layer wrapper
            subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask,
                                                                      position_ids, position_embeddings)
            # whether to prune by dynamic sparsity-->get subset layer sparsity-->
            if self.args.dsm == 'owl':
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm, self.args.granularity)
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    # self.logger.info(f"origin layer total sparsity ratio:{self.args.sparsity_ratio*len(self.all_blocks_ratio)}->adjustment layer {i} total sparsity ratio:{len(self.all_blocks_ratio)-self.all_blocks_ratio.sum()}")
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm,
                                                                           self.args.granularity)
                    self.all_layers_ratio.append(self.out_ratio_layer)
                else:
                    raise 
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps 
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
            elif self.args.dsm == 'mezo': # 逐层获取稀疏度
                self.layer_r = self.mezo_dsm(self.args.sparsity_ratio, inps, outs, attention_mask, position_ids, 
                                             position_embeddings, layer, subset, 
                                             zo_eps=self.args.zo_eps, epochs=self.args.epochs, 
                                             lr=self.args.lr, n_spsa=1, r_min=0.1, r_max=0.98)
                self.mezo_layer_r.append(self.layer_r.cpu().numpy().tolist())
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps 
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                self.logger.info(f"MEZO layer {i} dynamic sparsity ratios:{self.layer_r.cpu().numpy().tolist()}")
            elif self.args.dsm == "owl-mezo": # 先layer,再block
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, 'owl', 'per-block')
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    # self.logger.info(f"origin layer total sparsity ratio:{self.args.sparsity_ratio*len(self.all_blocks_ratio)}->adjustment layer {i} total sparsity ratio:{len(self.all_blocks_ratio)-self.all_blocks_ratio.sum()}")
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, 'owl', 'per-layer')
                    self.all_layers_ratio.append(self.out_ratio_layer)
                elif self.args.granularity == 'uniform':
                    del layer, subset, gpts, wrapped_layers
                    gc.collect()
                    torch.cuda.empty_cache()
                    if self.args.free:
                        self.layers[i].to("cpu")
                        torch.cuda.empty_cache()
                    continue
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps 
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                
            else: # uniform sparsity
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
        # print(self.all_layers_ratio, self.all_layers_blocks_ratio)
        
        if self.args.dsm == 'owl' or self.args.dsm == 'mezo':
            # print(self.all_layers_blocks_ratio)
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer' and self.args.dsm == 'owl':
                self.logger.info(self.all_layers_ratio)
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                            1 - self.args.sparsity_ratio)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                    layer.to(dev)
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                start = time.time()
                # 1. forward layer wrapper
                subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids,position_embeddings)

                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)

                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps

                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.model.model.layers[i].to("cpu")
                    torch.cuda.empty_cache()
        elif self.args.dsm == 'owl-mezo':
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                            1 - self.args.sparsity_ratio)
                self.logger.info(f"First, owl all layer sparsity ratio: {(1-self.all_layers_ratio).tolist()}")
            elif self.args.granularity == 'per-block':   
                self.logger.info(f"First, owl for layer blocks sparsity ratio: {(1-np.array(self.all_layers_blocks_ratio)).tolist()}")
            elif self.args.granularity == 'uniform':   
                self.logger.info(f"First, owl all layer sparsity ratio: {self.args.sparsity_ratio}")
            else:
                raise
            
            # Second: MeZO-->weight sparsity
            self.mezo_layer_r= []
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_cdevice_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                start = time.time()
                # 1. forward layer wrapper
                subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids,position_embeddings)
                # 1. forward layer wrapper
                # subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids)
                if self.args.granularity == 'per-layer':
                    self.layer_r = self.mezo_dsm(1-self.all_layers_ratio[i], inps, outs, attention_mask, position_ids, 
                                                 position_embeddings, layer, subset, 
                                                 zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, n_spsa=1, r_min=0.1, r_max=0.98)
                    
                elif self.args.granularity == 'per-block':
                    self.layer_r = self.mezo_dsm((1-np.array(self.all_layers_blocks_ratio))[i].tolist(), inps, outs, 
                                                 attention_mask,position_ids, position_embeddings, layer,subset, zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, n_spsa=1,r_min=0.1, r_max=0.98)
                elif self.args.granularity == 'uniform':
                    self.layer_r = self.mezo_dsm(self.args.sparsity_ratio, inps, outs, 
                                                 attention_mask,position_ids, position_embeddings, layer,subset, zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, n_spsa=1,r_min=0.1, r_max=0.98)
                self.mezo_layer_r.append(self.layer_r.cpu().numpy().tolist())
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                self.logger.info(f"Second, MEZO layer {i} dynamic sparsity ratios:{self.layer_r.cpu().numpy().tolist()}")
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                try:
                    # 如果当前模式是 mezo 或 owl-mezo 且已算出该层 r
                    if hasattr(self, "layer_r"):
                        r_layer = torch.tensor(self.layer_r, device="cpu", dtype=torch.float32).flatten()
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_mean", float(r_layer.mean().item()), i)
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_min", float(r_layer.min().item()),  i)
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_max", float(r_layer.max().item()),  i)
                        self._tb_add("add_histogram", f"mezo_summary/layer_{i}/r_hist", r_layer.numpy(), i)
                except Exception:
                    pass
            
            # 开始剪枝    
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_cdevice_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                    
                start = time.time()
                # 1. forward layer wrapper
                subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings)
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                    
        self.model.config.use_cache = self.use_cache
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")
        if getattr(self, "tb_writer", None):
            self.tb_writer.flush()
            # 如需长作业持续写入，可不 close。一次性脚本建议 close。
            # self.tb_writer.close()
        # return self.model
    
    
class Prune_LLAMA:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.nsamples = args.nsamples
        self.device = args.device

        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger


    def init_model(self): # share
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.layers
        # # TEST = False
        # TEST = True
        # if TEST:
        #     self.layers= self.layers[:2]
        
    def init_tensorboard(self):
    # --- TensorBoard: 可通过 args 开关/自定义日志目录 ---
        self.tb_enabled = getattr(self.args, "tb_enabled", True)
        self.tb_logdir  = getattr(self.args, "tb_logdir", f"{self.args.output_dir}/runs")
        self.tb_runname = getattr(self.args, "tb_runname", f"{self.args.exp_name}")
        self.tb_writer  = getattr(self.args, "tb_writer", None)
        if self.tb_enabled and (self.tb_writer is None):
            # 每个进程/实验独立 run name（可按需改成带 model 名）
            import datetime, os
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.tb_logdir, f"{self.tb_runname}-{ts}")
            self.tb_writer = SummaryWriter(log_dir=run_dir)

        # 便捷 no-op 包装，避免到处判空
        def _tb_add(fn_name, *a, **kw):
            w = self.tb_writer
            if w is None: 
                return
            getattr(w, fn_name)(*a, **kw)
        self._tb_add = _tb_add

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params
    @staticmethod
    def check_outlier_mean(mask, threshold):
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    @staticmethod
    def check_outlier_quantile(mask, quantile=0.99):
        W = mask.flatten()
        # 计算指定分位数
        q_value = torch.quantile(W, quantile)
        # 统计大于该分位数的元素
        count = (W > q_value).sum().item()
        total_params = W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio

    @staticmethod
    def check_outlier_normal(mask, k=3.0):
        W = mask.flatten()
        mean = torch.mean(W)
        std = torch.std(W)
        threshold = mean + k * std
        count = (W > threshold).sum().item()
        total_params = W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio
    
    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, subset, gpts, dsm='owl', granularity='per-block'):
        """
        Sparsity compensation
        Compensate for over-pruning caused by uniform sparsity due to different layer sensitivities, and balance sparsity.
        :param dsm:dynamic sparsity method-->global static adjustments
        :return:subset each layer sparsity
        """
        if dsm == "owl":
            # self.layer_wmetric = []
            # self.layer_outlier_ratios = {}
            if granularity == 'per-block':
                self.layer_outlier_ratios = []
                self.block_sizes = []
                for name in subset:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(gpts[name].scaler_row.reshape((1, -1)))
                    # SVD 
                    # U, S, VT = torch.linalg.svd(W_metric, full_matrices=False) 
                    # self.logger.info(f"{name} W_metric svd U shape {U.shape}")
                    # self.logger.info(f"{name} W_metric svd S shape {S.shape}")
                    # self.logger.info(f"{name} W_metric svd VT shape {VT.shape}")
                    # self.logger.info(f"{name} S MIN {S.min()}, MAX {S.max()}, MEAN {S.mean()}, MEDIAN {S.median()}")
                    
                    # # calculate block outlier ratio
                    block_outlier_ratio = self.check_outlier_mean(torch.flatten(W_metric.cpu()), self.args.Hyper_m) # why each has the same Hyper_m
                    # block_outlier_ratio = self.compute_sensitivity(S, top_ratio=0.2, mode="abs")
                    
                    # block_outlier_ratio = self.check_outlier_mean(torch.flatten(S.cpu()), self.args.Hyper_m) #
                    self.layer_outlier_ratios.append(block_outlier_ratio)
                    self.block_sizes.append(subset[name].weight.numel())  
                total_params = sum(self.block_sizes)
                block_weights = np.array(self.block_sizes) / total_params
                self.all_blocks_ratio = np.array(self.layer_outlier_ratios)
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / (self.all_blocks_ratio.max() - self.all_blocks_ratio.min())
                # [target_sparsity - lambda, target_sparsity + lambda] 
                target_sparsity = self.args.sparsity_ratio
                delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
                self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)  

                # 3. WEIGHTED CALIBRATION: ensure layer sparsity strictly equals target
                current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
                scale = target_sparsity / current_weighted_sparsity
                self.all_blocks_ratio =1-np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

                self.logger.info(f"Block sparsity: {1-self.all_blocks_ratio}, "
                                 f"Block outlier ratio: {self.all_blocks_ratio}, "
                                 f"Target sparsity: {target_sparsity:.4f}, "
                                 f"Weighted sparsity: {np.sum((1-self.all_blocks_ratio) * block_weights):.4f}, ")
                return self.all_blocks_ratio
            elif granularity == 'per-layer':
                self.layer_wmetric = []
                for name in subset:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(gpts[name].scaler_row.reshape((1, -1)))
                    self.layer_wmetric.append(torch.flatten(W_metric.cpu()))
                self.layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in self.layer_wmetric])  # [total weight number]
                self.out_ratio_layer = self.check_outlier_mean(self.layer_wmetric, self.args.Hyper_m)
                return self.out_ratio_layer
            
# === PATCH 1: helper functions (add inside class Prune_OPT) ===
    def _project_to_constraint(self, r, N, p_global):
        # r, N: 1D tensors on same device
        alpha = (N @ r - p_global * N.sum()) / (N @ N + 1e-12)
        return r - alpha * N
    
    def _sample_z_tangent(self, N):
        # gaussian then project to tangent space of N^T r = const
        self._seed_all(self.args.seed)
        n = N.numel()
        z = torch.randn(n, device=N.device, dtype=N.dtype)
        # z = torch.normal(mean=0, std=1, size=N.size(), device=N.device, dtype=N.dtype)
        # # project to {v | N^T v = 0}
        z = z - ( (N @ z) / (N @ N + 1e-12) ) * N
        # # normalize to unit l2
        z = z / (z.norm() + 1e-12)
        return z

    @staticmethod
    def _seed_all(seed: int):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)    
    def _compute_local_bounds(self, global_sparsity, layer_idx, num_layers, r_min, r_max):
        gs = float(global_sparsity)

        # 原先的带宽
        base_band = max(0.10, 0.30 * min(gs, 1.0 - gs))  # 稍放宽
        if num_layers <= 1:
            scale = 1.0
        else:
            pos = layer_idx / max(1, num_layers - 1)
            scale = 1.0 if pos < 1/3 else (0.9 if pos < 2/3 else 0.8)

        band = base_band * scale
        r_lo = max(r_min, gs - band)
        r_hi = min(r_max, gs + band)

        # —— 最小可探索带宽兜底（关键） ——
        MIN_BAND = 0.05  # 至少允许 ±5%
        if r_hi - r_lo < MIN_BAND:
            mid = min(r_max, max(r_min, gs))
            r_lo = max(r_min, mid - MIN_BAND/2)
            r_hi = min(r_max, mid + MIN_BAND/2)
            if r_hi <= r_lo:  # 再次兜底
                r_lo = max(r_min, gs - 0.03)
                r_hi = min(r_max, gs + 0.03)

        return float(r_lo), float(r_hi)
    
# === PATCH 2: replace your mezo_dsm with this version ===
    def mezo_dsm(self, global_sparsity, inps, outs, attention_mask, position_ids, position_embeddings, layer,
                subset, GPT, zo_eps=2e-2, epochs=1, lr=2e-1, n_spsa=1,
                r_min=0.10, r_max=0.98):
        """
        MeZO-DSM on per-Block sparsity vector r  (q/k/v/o, gate/up/down...)
        Key changes:
        - tangent-space noise under N-weighted global constraint
        - CRN: same seed & identical ADMM settings for L+/L-
        - consistent denominator (2*eps_r)
        - Euclidean projection to constraint + box clamp
        - simple precondition by 1/(N_i / meanN)
        """
        
        names = list(subset.keys())
        n = len(names)
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        dtype = torch.float32
        # —— 计算“本层”的自适应上下界（围绕 global_sparsity）——
        # ---- 在 mezo_dsm 的变量初始化后加上（如果没有的话）----
        if not hasattr(self, "_r_momentum"):
            self._r_momentum = None
        try:
            layer_idx = int(str(self.index_layer).split('_')[-1])
            num_layers = len(self.layers)
        except Exception:
            layer_idx, num_layers = 0, 1

        r_min_local, r_max_local = self._compute_local_bounds(
            global_sparsity=self.args.sparsity_ratio if isinstance(global_sparsity, list) else global_sparsity,
            layer_idx=layer_idx,
            num_layers=num_layers,
            r_min=r_min, r_max=r_max
        )
        # parameter counts per sub-layer
        with torch.no_grad():
            N = torch.tensor([subset[name].weight.numel() for name in names],
                            device=device, dtype=dtype)
            if isinstance(global_sparsity, list): # block sparsity
                r = torch.tensor(global_sparsity, device=device, dtype=dtype)
            else:
                r = torch.full((n,), float(global_sparsity), device=device, dtype=dtype)
            if isinstance(global_sparsity, list): # block sparsity
                r = self._project_to_constraint(r, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
            else:
                r = self._project_to_constraint(r, N, global_sparsity).clamp_(r_min_local, r_max_local)
            # print(f"layer {layer_idx} initial sparsity: {r.detach().cpu().tolist()}")
            self.logger.info(f"layer {layer_idx} initial sparsity: {r.detach().cpu().tolist()} "
                             f"r(min/mean/max)=({r.min().item():.3f}/{r.mean().item():.3f}/{r.max().item():.3f}) "
                             f"weighted={float((r*N).sum()/N.sum()):.4f}")
            
        # === 在 mezo_dsm() 里，计算出 layer_idx 之后，定义一个 tensorboard 基础 tag ===
        base_tag = f"mezo/layer_{layer_idx}"

        # （可选）记录初始化 r 与 dense baseline
        try:
            # 轻量：仅统计/直方图，不写大张量
            self._tb_add("add_scalar", f"{base_tag}/init_weighted_r", float((r*N).sum()/N.sum()), 0)
            self._tb_add("add_scalar", f"{base_tag}/r_min", float(r.min().item()), 0)
            self._tb_add("add_scalar", f"{base_tag}/r_max", float(r.max().item()), 0)
            self._tb_add("add_histogram", f"{base_tag}/r_hist", r.detach().cpu().numpy(), 0)
        except Exception:
            pass

        try:
            init_loss = eval_avg_loss(r, max_eval_samples=min(16, inps.shape[0]), seed_for_all=self.args.seed)
            self._tb_add("add_scalar", f"{base_tag}/epoch_loss", float(init_loss), 0)
        except Exception:
            init_loss = None
            

        # precompute dense outputs for CRN (already done outside, keep your logic)
        dense_Y = torch.zeros_like(outs)
        for j in range(inps.shape[0]):
            with torch.no_grad():
                # dense_Y[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

                dense_Y[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

        # ---- inner: loss after pruning this block with sparsity r ----
        def prune_and_loss(current_r, X, Y, seed_for_all):
            # CRN: same seeds & deterministic flags
            self._seed_all(seed_for_all)
            # copy the block & collect stats
            layer_copy = copy.deepcopy(layer)
            # subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, GPT)
            if self.args.use_wanda_forward:
                if self.args.d2_wanda:
                    subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, position_embeddings, GPT=D2Wanda)
                else:
                    subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, position_embeddings, GPT=Wanda)
            else:
                subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, position_embeddings, GPT)
            # 目标全局稀疏度（标量），兼容 list 情况
            if isinstance(global_sparsity, list):
                gvec = torch.tensor(global_sparsity, device=device, dtype=dtype)
                p_target = (gvec * N).sum() / (N.sum() + 1e-12)
            else:
                p_target = torch.tensor(float(global_sparsity), device=device, dtype=dtype)
            # apply pruning per sub-layer
            for i, name in enumerate(subset_copy):
                sp = float(current_r[i].item())
                if not math.isfinite(sp):
                    sp = float(global_sparsity)   # 或者 (r_min_local + r_max_local)/2
                sp = float(min(r_max_local, max(r_min_local, sp)))
                if self.args.use_wanda_forward:
                    gpts[name].fasterprune(sp, self.prune_n, self.prune_m)
                else:
                    try:
                        if self.args.use_wanda_forward:
                            gpts[name].fasterprune(sp, self.prune_n, self.prune_m)
                        else:
                            if self.args.prune_method == 'sparsegpt':
                                gpts[name].fasterprune(sp, self.prune_n, self.prune_m, blocksize=128, percdamp=.01)
                            elif self.args.prune_method == 'wanda':
                                gpts[name].fasterprune(sp, self.prune_n, self.prune_m)
                            elif self.args.prune_method == 'pruner-zero':
                                indexed = f'{name}_{self.index_layer}'
                                gradients = self.gradients_l2[indexed]
                                gpts[name].fasterprune(sp, self.prune_n, self.prune_m, gradients, engine=self.engine)
                            elif self.args.prune_method == 'admm-grad':
                                # 固定 ADMM 配置，确保 L+/L- 完全一致
                                gpts[name].fasterprune(sp, self.prune_n, self.prune_m, percdamp=.1,
                                                    iterative_prune=15, iters=20, per_out=False)
                            else:
                                raise NotImplementedError
                        gpts[name].free()
                    except Exception:
                        # 一旦剪枝内部抛错（含 NaN/Inf），直接返回极大损失，避免污染本轮估计
                        layer_copy.cpu(); del layer_copy, subset_copy, gpts
                        torch.cuda.empty_cache()
                        return torch.tensor(1e6, device=X.device, dtype=torch.float32), torch.tensor(1e6, device=X.device, dtype=torch.float32), torch.tensor(1e6, device=X.device, dtype=torch.float32)
                
            recon_loss = 0
            sparsity_reg_loss = 0
            with torch.no_grad():
                sparse_Y = layer_copy(X, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # 若输出存在非有限值，返回极大损失
                if not torch.isfinite(sparse_Y).all():
                    loss = torch.tensor(1e6, device=X.device, dtype=torch.float32)
                else:
                    den = float(Y.var().item())
                    if not math.isfinite(den) or den < 1e-8:
                        den = 1e-6
                    loss = torch.nn.functional.mse_loss(sparse_Y, Y) / (den+1e-6)

                # 稀疏度正则化
                if getattr(self.args, "sparse_reg", True):
                    recon_loss += loss
                    # ========== 新增：L2 正则 ==========

                    # 全局加权稀疏度与目标的偏差
                    p_now = (current_r * N).sum() / (N.sum() + 1e-12)
                    global_reg = (p_now - p_target.to(p_now.dtype))**2

                    lam_g = float(getattr(self.args, "lambda_global", 0.0))
                    sparsity_reg_loss += lam_g * global_reg

                    loss = recon_loss + sparsity_reg_loss
                    # self.sparsity_reg_loss = lam_g * global_reg
                    
            # 清理
            layer_copy.cpu(); del layer_copy, subset_copy, gpts
            torch.cuda.empty_cache()
            return loss, recon_loss, sparsity_reg_loss
        # ====== 评估函数：给定 r，返回在一小批输入上的平均层内 MSE ======
        def eval_avg_loss(current_r, max_eval_samples= min(16, inps.shape[0]), seed_for_all=self.args.seed):
            self._seed_all(seed_for_all)
            # 子采样以加速（也可用全体 inps）
            idxs = list(range(inps.shape[0]))[:max_eval_samples]
            losses = []
            for j_ in idxs:
                X_j = inps[j_].unsqueeze(0); Y_j = dense_Y[j_].unsqueeze(0)
                # L = prune_and_loss(current_r, X_j, Y_j, seed_for_all)
                L,_,_ = prune_and_loss(current_r, X_j, Y_j, seed_for_all)
                losses.append(float(L.item()))
            return float(np.mean(losses))
        
        loss_best = eval_avg_loss(r, max_eval_samples=min(16, inps.shape[0]), seed_for_all=self.args.seed)
        r_best = r.clone()
        no_improve = 0
        patience = 2            # 可调：2~4


        meanN = N.mean()
        precond = (N / meanN).clamp(min=1e-6)

        # ---- MeZO loop ----
        # meanN = N.mean()
        # precond = (N / meanN).clamp(min=1e-6)  # scale for gradient; used as divisor
        steps = 0
        prev_epoch_loss = 1e3
        for epoch in range(epochs):
            total_loss = 0.0
            total_r_loss = 0.0
            total_s_loss = 0.0
            # === “学不动”时的临时加速器 ===
            lr_epoch, zo_eps_epoch, n_spsa_epoch = lr, zo_eps, n_spsa
            if prev_epoch_loss is not None:
                improve = prev_epoch_loss - 0.0  # 用上一轮均值初始化
            # ======= Batch version of SPSA gradient estimate =======
            batch_size = self.args.batch_size  # 可以设为 self.args.batch_mezo
            num_batches = math.ceil(inps.shape[0] / batch_size)
            
            for batch_idx in range(num_batches):
                batch_slice = slice(batch_idx * batch_size, min((batch_idx+1)*batch_size, inps.shape[0]))
                X_batch = inps[batch_slice]
                Y_batch = dense_Y[batch_slice]

                g_hat = torch.zeros_like(r)
                loss_plus_acc, loss_minus_acc = 0.0, 0.0
                r_loss_plus_acc, s_loss_plus_acc = 0.0, 0.0

                for _ in range(n_spsa_epoch):
                    z = self._sample_z_tangent(N)
                    eps_r = float(zo_eps_epoch)

                    if isinstance(global_sparsity, list):
                        r_plus  = self._project_to_constraint(r + eps_r*z, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                        r_minus = self._project_to_constraint(r - eps_r*z, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                    else:
                        r_plus  = self._project_to_constraint(r + eps_r*z, N, global_sparsity).clamp_(r_min_local, r_max_local)
                        r_minus = self._project_to_constraint(r - eps_r*z, N, global_sparsity).clamp_(r_min_local, r_max_local)

                    # L_plus_mean = prune_and_loss(r_plus,  X_batch, Y_batch, self.args.seed)
                    # L_minus_mean = prune_and_loss(r_minus, X_batch, Y_batch, self.args.seed)
                    L_plus_mean, r_plus_mean, s_plus_mean = prune_and_loss(r_plus,  X_batch, Y_batch, self.args.seed)
                    L_minus_mean, _, _ = prune_and_loss(r_minus, X_batch, Y_batch, self.args.seed)
                    
                    # L_plus_mean = prune_and_loss(r + eps_r*z,  X_batch, Y_batch, self.args.seed)
                    # L_minus_mean = prune_and_loss(r - eps_r*z, X_batch, Y_batch, self.args.seed)
                    


                    coeff = (L_plus_mean - L_minus_mean) / (2.0 * eps_r + 1e-12)
                    g_hat += coeff * z
                    
                    loss_plus_acc += L_plus_mean
                    loss_minus_acc += L_minus_mean
                    
                    r_loss_plus_acc += r_plus_mean
                    s_loss_plus_acc += s_plus_mean
                    
                    


                # g_hat /= float(n_spsa)
                g_hat /= float(n_spsa_epoch)
                total_loss += loss_plus_acc / max(1, n_spsa_epoch)
                
                total_r_loss += r_loss_plus_acc / max(1, n_spsa_epoch)
                total_s_loss += s_loss_plus_acc / max(1, n_spsa_epoch)
                
                # total_loss += loss_plus_acc / max(1, _n)
                # ---- 动量更新 + 投影 ----
                if self._r_momentum is None or self._r_momentum.shape != g_hat.shape:
                    self._r_momentum = torch.zeros_like(g_hat)
                beta = 0.9
                self._r_momentum = beta * self._r_momentum + (1 - beta) * (g_hat / ((N / N.mean()).clamp(min=1e-6)))
                # r = r - lr_epoch * self._r_momentum
                r = r - lr_epoch * g_hat
                    
                if isinstance(global_sparsity, list): # block sparsity
                    r = self._project_to_constraint(r, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)
                    r = torch.nan_to_num(r, nan=self.args.sparsity_ratio,
                                        posinf=r_max_local, neginf=r_min_local)
                else:
                    r = self._project_to_constraint(r, N, global_sparsity).clamp_(r_min_local, r_max_local)
                    r = torch.nan_to_num(r, nan=float(global_sparsity),
                                        posinf=r_max_local, neginf=r_min_local)
                steps += 1
                
            # ====== 统计改进，若几乎不降就“踹一脚” ======
            epoch_loss_mean = total_loss / num_batches

            epoch_r_loss_mean = total_r_loss / num_batches
            epoch_s_loss_mean = total_s_loss / num_batches
            
            if prev_epoch_loss is not None:
                improve = prev_epoch_loss - epoch_loss_mean
                if improve < 1e-5:
                    lr_epoch = lr * 1.5
                    zo_eps_epoch = zo_eps * 1.5
                    n_spsa_epoch = max(1, n_spsa * 2)
                else:
                    lr_epoch, zo_eps_epoch, n_spsa_epoch = lr, zo_eps, n_spsa
                if improve > 1e-5:
                    self.logger.info(f"current r is positive adjustment, please update")
                    prev_epoch_loss = epoch_loss_mean
                    r_best = r
                else:
                    self.logger.info(f"current r is negative adjustment, don't update")
                    no_improve += 1
                    if no_improve >= patience:
                        self.logger.info(f"[MEZO] early stop at epoch {epoch}, best_loss={prev_epoch_loss:.6f}")
                        break
            try:
                self._tb_add("add_scalar", f"{base_tag}/epoch_loss", float(epoch_loss_mean), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/loss_best", float(prev_epoch_loss), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/weighted_r", float((r*N).sum()/N.sum()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_min", float(r.min().item()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_max", float(r.max().item()), epoch+1)
                self._tb_add("add_scalar", f"{base_tag}/r_mean", float(r.mean().item()), epoch+1)
                if self.args.sparse_reg:
                    self._tb_add("add_scalar", f"{base_tag}/epoch_reg_loss", float(epoch_r_loss_mean), epoch+1)
                    self._tb_add("add_scalar", f"{base_tag}/epoch_sparse_loss", float(epoch_s_loss_mean), epoch+1)
                # 直方图别太频繁；每若干 epoch 或最后一轮再记也行
                if (epoch % 5 == 0) or (epoch+1 == epochs):
                    self._tb_add("add_histogram", f"{base_tag}/r_hist", r.detach().cpu().numpy(), epoch+1)
            except Exception:
                pass

            self.logger.info(f"[MEZO] epoch={epoch} loss={epoch_loss_mean:.6f} "
                            f"r(min/mean/max)=({r.min().item():.3f}/{r.mean().item():.3f}/{r.max().item():.3f}) "
                            f"weighted={float((r*N).sum()/N.sum()):.4f}")

            # 手动轻微衰减学习率（可选）
            # lr *= 0.95

        if isinstance(global_sparsity, list): # block sparsity
            r = self._project_to_constraint(r_best, N, self.args.sparsity_ratio).clamp_(r_min_local, r_max_local)   
        else:
            r = self._project_to_constraint(r_best, N, global_sparsity).clamp_(r_min_local, r_max_local)
        if isinstance(global_sparsity, list): # block sparsity
            r = torch.nan_to_num(r, nan=float(self.args.sparsity_ratio),
                                posinf=r_max_local, neginf=r_min_local)
        else:
            r = torch.nan_to_num(r, nan=float(global_sparsity),
                                posinf=r_max_local, neginf=r_min_local)
        return r.detach().cpu()


    def greedy_dsm(
        self,
        global_sparsity,                  # 目标加权稀疏度（如 0.5/0.6/0.7/0.8）
        inps, outs, attention_mask, position_ids, position_embeddings, layer,
        subset, GPT,
        alpha=0.02,                       # 基础步长系数，越大越快但更粗糙
        r_min=0.10, r_max=0.98,           # 每个 block 的下/上界
        eval_batch=16,                    # 每步评估用多少样本
        lam_reg=0.0,                      # 轻微正则（如 1e-4），抑制把预算全堆到一个块
        max_steps=10000,                 # 最大外部步数，防死循环
        seed=None
    ):
        """
        Block-wise Greedy Optimization (见你发的 Algorithm 1)
        返回：长度为 n 的 tensor，表示该层各子块的稀疏度 r_i
        """

        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        names = list(subset.keys())
        n = len(names)

        # --- 各子块参数规模 f_i 及总规模 F ---
        with torch.no_grad():
            f = torch.tensor([subset[name].weight.numel() for name in names], dtype=torch.float64, device=device)
        F = float(f.sum().item())

        # --- 初始化 p 与全局进度 P（按论文：p=0；此处给 r_min 起步更稳）---
        p = torch.full((n,), float(r_min), dtype=torch.float64, device=device)
        P = float((p * f).sum().item() / F)                   # 当前加权稀疏度
        target = float(global_sparsity)
        target_sum = target * F

        # --- 预计算 dense 输出（CRN）---
        dense_Y = torch.zeros_like(outs)
        for j in range(inps.shape[0]):
            with torch.no_grad():
                dense_Y[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, 
                                   position_embeddings=position_embeddings)[0]

        if seed is None: seed = getattr(self.args, "seed", 0)

        # --- 两个小工具：评估、一步剪枝并算损失 ---
        def _prune_and_loss(current_r, X, Y, seed_for_all):
            self._seed_all(seed_for_all)
            layer_copy = copy.deepcopy(layer)
            subset_copy, gpts = self.forward_layer_wrapper(layer_copy, X, Y, attention_mask, position_ids, 
                                                           position_embeddings, GPT)

            for i, name in enumerate(subset_copy):
                sp = float(current_r[i])
                if self.args.prune_method == 'sparsegpt':
                    gpts[name].fasterprune(sp, self.prune_n, self.prune_m, blocksize=128, percdamp=.01)
                elif self.args.prune_method == 'wanda':
                    gpts[name].fasterprune(sp, self.prune_n, self.prune_m)
                elif self.args.prune_method == 'pruner-zero':
                    indexed = f'{name}_{self.index_layer}'
                    gradients = self.gradients_l2[indexed]
                    gpts[name].fasterprune(sp, self.prune_n, self.prune_m, gradients, engine=self.engine)
                elif self.args.prune_method == 'admm-grad':
                    gpts[name].fasterprune(sp, self.prune_n, self.prune_m, percdamp=.1, iterative_prune=15, iters=20, per_out=False)
                else:
                    raise NotImplementedError
                gpts[name].free()

            with torch.no_grad():
                sparse_Y = layer_copy(X, attention_mask=attention_mask, position_ids=position_ids, 
                                      position_embeddings=position_embeddings)[0]
                # 标准化 MSE，抵消不同层/样本的尺度差
                loss = torch.nn.functional.mse_loss(sparse_Y, Y) / (Y.var() + 1e-8)

            layer_copy.cpu(); del layer_copy, subset_copy, gpts
            torch.cuda.empty_cache()
            return float(loss.item())

        def _eval_avg_loss(current_r, mb=eval_batch, seed_for_all=seed):
            idxs = list(range(inps.shape[0]))[:max(1, min(mb, inps.shape[0]))]
            losses = []
            for j in idxs:
                X_j = inps[j].unsqueeze(0); Y_j = dense_Y[j].unsqueeze(0)
                losses.append(_prune_and_loss(current_r, X_j, Y_j, seed_for_all))
            return float(np.mean(losses))

        # --- 基础步长：α · (F/f_i)（大块走小步，小块走大步），再加一个“均衡”折扣(1 + p_i) ---
        def _base_step(i):
            # 避免永远堆在同一个块：随着 p_i 增大，步长逐渐变小
            return float(alpha * (F / float(f[i])) / (1.0 + float(p[i])))

        # --- 贪心主循环 ---
        best_p = p.clone()
        best_loss = _eval_avg_loss(best_p)
        outer = 0

        # 若起步就已达标（极端 r_min 很大时），直接微调
        if (p * f).sum().item() >= target_sum - 1e-12:
            outer = max_steps  # 跳过循环

        while outer < max_steps and (p * f).sum().item() < target_sum - 1e-12:
            outer += 1

            # 逐个尝试：p_i 增加一步，记录误差
            E = np.full((n,), np.inf, dtype=np.float64)
            step_cache = np.zeros((n,), dtype=np.float64)

            cur_sum = float((p * f).sum().item())       # = P * F
            need_sum = max(0.0, target_sum - cur_sum)   # 还差多少“加权稀疏度”

            for i in range(n):
                if float(p[i]) >= r_max - 1e-12:   # 到上界就跳过
                    continue

                raw = _base_step(i)
                # 不超预算、不超上界的最大可走步长
                cap_by_budget = need_sum / float(f[i].item())
                cap_by_box    = float(r_max - p[i])
                step = max(1e-6, min(raw, cap_by_budget, cap_by_box))

                if step <= 1e-12:
                    continue

                p_try = p.clone()
                p_try[i] = float(min(r_max, max(r_min, float(p[i]) + step)))

                # 轻微正则（可为 0）
                loss_i = _eval_avg_loss(p_try)
                if lam_reg > 0.0:
                    loss_i += lam_reg * float(p_try[i])

                E[i] = loss_i
                step_cache[i] = step

            # 没有可更新的块，收工
            if not np.isfinite(E).any():
                self.logger.info("[GREEDY] all blocks saturated or no feasible step; stop.")
                break

            # 选误差最小的块 j
            j = int(np.nanargmin(E))
            # —— 真正执行更新（仍旧严守预算与边界）——
            cur_sum = float((p * f).sum().item())
            need_sum = max(0.0, target_sum - cur_sum)
            max_step = need_sum / float(f[j].item())
            step = max(1e-6, min(step_cache[j], max_step, float(r_max - p[j])))

            prev_pj = float(p[j])
            p[j] = float(min(r_max, max(r_min, prev_pj + step)))
            P = float((p * f).sum().item() / F)

            # —— 接受即记录（平台损失也更新 best_p）——
            best_loss = float(E[j])
            best_p = p.clone()

            self.logger.info(
                f"[GREEDY] step={outer} pick={names[j]} "
                f"P={P:.4f}/{target:.4f} "
                f"r(min/mean/max)=({float(p.min()):.3f}/{float(p.mean()):.3f}/{float(p.max()):.3f}) "
                f"weighted={float((p*f).sum().item()/F):.4f} best_loss={best_loss:.6f}"
            )

            # 这一步若已经精确打到目标，直接退出
            if target_sum - (p * f).sum().item() <= 1e-12:
                break

        # ===== 兜底：把 P 精确拉到 target =====
        cur_sum = float((p * f).sum().item())
        gap = target_sum - cur_sum
        if abs(gap) > 1e-9:
            # 优先调整最后一次更新的 j；找不到就挑容量最大的块
            candidate_idxs = list(range(n))
            candidate_idxs.sort(key=lambda t: -float(f[t]))  # 大块优先，微调影响小
            picked = None
            for t in [j] + candidate_idxs if 'j' in locals() else candidate_idxs:
                # 还能往上加
                if float(p[t]) < r_max - 1e-9 and gap > 0:
                    picked = t; break
                # 还能往下减（通常不会需要，因为起点是 r_min）
                if float(p[t]) > r_min + 1e-9 and gap < 0:
                    picked = t; break
            if picked is None:
                picked = int(np.argmax(f.cpu().numpy()))
            adj = gap / float(f[picked].item())
            p[picked] = float(min(r_max, max(r_min, float(p[picked]) + adj)))

        # 保底裁盒
        p.clamp_(r_min, r_max)

        # 返回（double->float，放回 CPU）
        r = p.to(torch.float32).detach().cpu()
        return r
    
    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        device = self.model.model.embed_tokens.weight.device  #
        if device.type == 'cpu':
            device = self.device
            self.model.model.embed_tokens.to(device)
        else:
            device = device.index
            self.model.model.embed_tokens.to(device)
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype  # torch.float16
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                           device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                cache["position_embeddings"] = kwargs['position_embeddings']
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:  #
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))  # batch[0]-->[1,2048]
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache["position_embeddings"]
        self.model.config.use_cache = self.use_cache  # True
        if self.args.free:
            self.model.model.embed_tokens.to("cpu")
        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids, position_embeddings

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids, position_embeddings, GPT):
        subset = self.find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = GPT(self.args, subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,768]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0] # [1,2048,768)
        for h in handles:
            h.remove()
        return subset, gpts

    @timeit
    def prune_layer_weight(self, subset, gpts):
        for i,name in enumerate(subset):
            if self.args.dsm == 'owl':
                if self.args.granularity == 'per-block':
                    # self.sparsity_ratio = 1-self.all_blocks_ratio[i]
                    self.sparsity_ratio = 1-self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    # self.sparsity_ratio = self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    self.logger.info(f"block sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
                elif self.args.granularity == 'per-layer':
                    self.sparsity_ratio = 1-self.all_layers_ratio[int(self.index_layer.split('_')[-1])]
                    self.logger.info(f"layer sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
            elif self.args.dsm == 'mezo' or self.args.dsm == 'owl-mezo':
                self.sparsity_ratio = self.mezo_layer_r[int(self.index_layer.split('_')[-1])][i]
            elif self.args.dsm == 'greedy':
                self.sparsity_ratio = self.greedy_layer_r[int(self.index_layer.split('_')[-1])][i]
                
            if self.args.prune_method == 'sparsegpt':
                self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m,
                                       blocksize=128, percdamp=.01)
                gpts[name].free()

            elif self.args.prune_method == 'wanda':
                self.logger.info(f"pruning {name} by Wanda")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()

            elif self.args.prune_method == 'pruner-zero':
                self.logger.info(f"pruning {name} by Pruner-Zero")
                indexed_name = f'{name}_{self.index_layer}'
                gradients = self.gradients_l2[indexed_name]
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, gradients, engine=self.engine)
                gpts[name].free()
                
            elif self.args.prune_method == 'admm-grad':
                self.logger.info(f"pruning {name} by ADMM-Grad")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, percdamp=.1, iterative_prune=15, iters=20, per_out=False)
                gpts[name].free()

            else:
                raise NotImplementedError
            torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
        if self.args.prune_method == 'pruner-zero':
            self.logger.info("you must loading model gradient for pruner-zero")
            self.gradients_l2 = self.args.gradients_l2
            self.engine = self.args.engine   # GPTree.load_tree('../Pruner-Zero/data/best_tree.json')
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []
        self.mezo_layer_r= []
        self.greedy_layer_r = []
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)

            elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                dev = self.device
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)

            start = time.time()
            # 1. forward layer wrapper
            if self.args.prune_method == 'sparsegpt':
                GPT = SparseGPT
            elif self.args.prune_method == 'wanda':
                GPT = Wanda
            elif self.args.prune_method == 'pruner-zero':
                GPT = PrunerZero
            elif self.args.prune_method == 'admm-grad':
                GPT = AdmmGrad
            else:
                raise NotImplementedError

            # 1. forward layer wrapper
            subset, gpts= self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings, GPT)
            # whether to prune by dynamic sparsity-->get subset layer sparsity-->
            if self.args.dsm == 'owl':
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, self.args.dsm, self.args.granularity)
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    # self.logger.info(f"origin layer total sparsity ratio:{self.args.sparsity_ratio*len(self.all_blocks_ratio)}->adjustment layer {i} total sparsity ratio:{len(self.all_blocks_ratio)-self.all_blocks_ratio.sum()}")
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, self.args.dsm,
                                                                           self.args.granularity)
                    self.all_layers_ratio.append(self.out_ratio_layer)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
            elif self.args.dsm == 'mezo': # 逐层获取稀疏度
                self.layer_r = self.mezo_dsm(self.args.sparsity_ratio, inps, outs, attention_mask, position_ids, 
                                             position_embeddings, layer, subset, GPT, 
                                             zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, 
                                             n_spsa=1,r_min=0.1, r_max=0.98)
                self.mezo_layer_r.append(self.layer_r.cpu().numpy().tolist())
                self.logger.info(f"MEZO layer {i} dynamic sparsity ratios:{self.layer_r.cpu().numpy().tolist()}")
                
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
            elif self.args.dsm == "owl-mezo": # 先layer,再block
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, 'owl', 'per-block')
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, 'owl', 'per-layer')
                    self.all_layers_ratio.append(self.out_ratio_layer)
                    
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
            elif self.args.dsm == 'greedy':
                self.layer_r = self.greedy_dsm(
                    global_sparsity=self.args.sparsity_ratio,
                    inps=inps, outs=outs,
                    attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings,  
                    layer=layer, subset=subset, GPT=GPT,
                    alpha=0.02,              # 建议起点：0.01~0.03
                    r_min=0.10, r_max=0.98,  # 与 MeZO 保持一致
                    eval_batch=min(16, self.nsamples),
                    lam_reg=1e-4,
                    max_steps=10000,
                    seed=self.args.seed
                )
                self.greedy_layer_r.append(self.layer_r.cpu().numpy().tolist())
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                self.logger.info(f"GREEDY layer {i} dynamic sparsity ratios:{self.layer_r.cpu().numpy().tolist()}")
                
            else:
                self.prune_layer_weight(subset, gpts)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps   
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                    
        if self.args.dsm == 'owl' or self.args.dsm == 'mezo' or self.args.dsm == 'greedy':
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                            1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (1 - self.args.sparsity_ratio)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                

                start = time.time()
                # 1. forward layer wrapper
                if self.args.prune_method == 'sparsegpt':
                    GPT = SparseGPT
                elif self.args.prune_method == 'wanda':
                    GPT = Wanda
                elif self.args.prune_method == 'pruner-zero':
                    GPT = PrunerZero
                else:
                    raise NotImplementedError
                # 1. forward layer wrapper
                subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings, GPT)
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
        elif self.args.dsm == 'owl-mezo':
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                            1 - self.args.sparsity_ratio)
                self.logger.info(f"First, owl all layer sparsity ratio: {(1-self.all_layers_ratio).tolist()}")
            elif self.args.granularity == 'per-block':   
                self.logger.info(f"First, owl for layer blocks sparsity ratio: {(1-np.array(self.all_layers_blocks_ratio)).tolist()}")
            # Second: MeZO-->weight sparsity
            self.mezo_layer_r= []
            
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                start = time.time()
                # 1. forward layer wrapper
                if self.args.prune_method == 'sparsegpt':
                    GPT = SparseGPT
                elif self.args.prune_method == 'wanda':
                    GPT = Wanda
                elif self.args.prune_method == 'pruner-zero':
                    GPT = PrunerZero
                elif self.args.prune_method == 'admm-grad':
                    GPT = AdmmGrad
                else:
                    raise NotImplementedError
                # 1. forward layer wrapper
                subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings, GPT)
                # mezo确定各个bclok的动态稀疏度
                if self.args.granularity == 'per-layer':
                    self.layer_r = self.mezo_dsm(1-self.all_layers_ratio[i], inps, outs, attention_mask, position_ids,
                                                 position_embeddings, layer,subset, GPT, 
                                                 zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, n_spsa=1, r_min=0.1, r_max=0.98)
                elif self.args.granularity == 'per-block':
                    self.layer_r = self.mezo_dsm((1-np.array(self.all_layers_blocks_ratio))[i].tolist(), inps, outs, 
                                                 attention_mask, position_ids,position_embeddings, layer,subset, GPT, 
                                                 zo_eps=self.args.zo_eps, epochs=self.args.epochs, lr=self.args.lr, n_spsa=1, r_min=0.1, r_max=0.98)
                self.mezo_layer_r.append(self.layer_r.cpu().numpy().tolist())
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                inps, outs = outs, inps
                self.logger.info(f"MEZO layer {i} dynamic sparsity ratios:{self.layer_r.cpu().numpy().tolist()}")
                del layer, subset, gpts
                
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                try:
                    # 如果当前模式是 mezo 或 owl-mezo 且已算出该层 r
                    if hasattr(self, "layer_r"):
                        r_layer = torch.tensor(self.layer_r, device="cpu", dtype=torch.float32).flatten()
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_mean", float(r_layer.mean().item()), i)
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_min", float(r_layer.min().item()),  i)
                        self._tb_add("add_scalar", f"mezo_summary/layer_{i}/r_max", float(r_layer.max().item()),  i)
                        self._tb_add("add_histogram", f"mezo_summary/layer_{i}/r_hist", r_layer.numpy(), i)
                except Exception:
                    pass
            # 开始剪枝
            inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                else:
                    dev = layer.self_attn.q_proj.weight.device 
                layer.to(dev)
                if attention_mask is None:
                    inps, outs, position_ids, position_embeddings = inps.to(dev), outs.to(dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                else:
                    inps, outs, attention_mask, position_ids, position_embeddings= inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev), position_embeddings.to(dev) if isinstance(position_embeddings, torch.Tensor) else tuple(x.to(dev) for x in position_embeddings)
                start = time.time()
                # 1. forward layer wrapper
                if self.args.prune_method == 'sparsegpt':
                    GPT = SparseGPT
                elif self.args.prune_method == 'wanda':
                    GPT = Wanda
                elif self.args.prune_method == 'pruner-zero':
                    GPT = PrunerZero
                elif self.args.prune_method == 'admm-grad':
                    GPT = AdmmGrad
                else:
                    raise NotImplementedError
                # 1. forward layer wrapper
                subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, position_embeddings, GPT)
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                        
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
                   
        self.model.config.use_cache = self.use_cache
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")
        if getattr(self, "tb_writer", None):
            self.tb_writer.flush()
            # 如需长作业持续写入，可不 close。一次性脚本建议 close。
            # self.tb_writer.close()
