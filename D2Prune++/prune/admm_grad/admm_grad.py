import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class AdmmGrad:

    def __init__(self, args, layer):
        self.args = args
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.shape[0]
        self.columns = layer.weight.shape[1]
        self.XX = torch.zeros((self.columns, self.columns), device=self.dev)
        if self.args.dsm !=None:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # if self.args.d2_admm:
        #     self.YY = torch.zeros((self.rows, self.rows), device=self.dev)
        #     self.nsamples = 0
        #     self.s = self.args.seq_len if self.args.auto_s else self.args.s
        #     self.r1 = self.args.r1
        #     self.r2 = self.args.r2

    def add_batch(self, inp, out):
        X = inp.reshape(-1, inp.shape[-1]).float()
        self.XX += X.T.matmul(X)
        # if self.args.d2_admm:
        #     Y = out.reshape(-1, out.shape[-1]).float()
        #     self.YY += Y.T.matmul(Y)

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, percdamp=.1, iterative_prune=15, iters=20, per_out=False
    ):
        XX = self.XX     
        norm = torch.diag(XX).sqrt() + 1e-8
        if self.args.d2_admm:
            beta = self.args.beta
            norm = beta * norm
        # if self.args.d2_admm: # W-->[D_out, D_in]
        #     YY = self.YY
        #     norm_y = (torch.diag(YY).sqrt() + 1e-8)[:, None]# [D_out, 1]
        #     norm_x = norm[None, :] # [1, D_in] 
        #     print(norm_y.shape)
        #     print(norm_x.shape)

        print(norm.min(), norm.max())
        XX = XX / norm
        XX = (XX.T / norm).T
        # beta = 0.9
        # if self.args.d2_admm: # sqart
        #     # scale_y = torch.norm(norm_y, p=2) / self.s
        #     # XX = XX / (norm[None, :] + self.r1 * (scale_y ** (1/2)) * (norm_x** (1/2)) - self.r2 * (norm_x ** (1)))
        #     # XX = (XX.T / (norm[None, :] + self.r1 * (scale_y ** (1/2)) * (norm_x** (1/2)) - self.r2 * (norm_x ** (1)))).T
        #     XX = XX / (beta * norm)
        #     XX = (XX.T / ( beta *norm)).T
        # else:
        #     XX = XX / norm
        #     XX = (XX.T / norm).T
        W = (self.layer.weight.float().detach() * norm).T  # w * ||x||2 -->wanda的metric
        # if self.args.d2_admm: # sqart
        #     # W = (self.layer.weight.float().detach()).T
        #     # W += self.r1 * ((norm_y ** (1/2)) * self.layer.weight.float().detach() * norm_x** (1/2)).T 
        #     # W += -self.r2 * (self.layer.weight.float().detach() * (norm_x ** (1))).T 
        #     W = (self.layer.weight.float().detach() * (beta * norm)).T

        rho0 = percdamp * torch.diag(XX).mean()
        diag = torch.arange(XX.shape[0], device=XX.device)
        XX[diag,diag] += rho0
#        XX = XX + torch.eye(XX.shape[0], device=XX.device)*rho0

        if iterative_prune == 0:
            if prune_n != 0:
                WT = W.T.reshape((W.shape[1]*W.shape[0]//4, 4)).abs()
                mask = torch.zeros_like(WT)
                sort_inds = WT.sort(dim=1)[1]
                mask[torch.arange(WT.shape[0]), sort_inds[:,2]] = 1
                mask[torch.arange(WT.shape[0]), sort_inds[:,3]] = 1
                #mask = (WT >= thres).reshape(W.T.shape).T
                mask = mask.reshape(W.T.shape).T
            elif per_out:
                thres = (W).abs().sort(dim=0)[0][int(W.shape[0] * sparsity)]
                mask = ((W).abs() >= thres.unsqueeze(0))
                del thres
            else:
#                thres = (W).abs().flatten().kthvalue(int(W.numel() * sparsity)+1)[0]
#                mask = ((W).abs() > thres)
                topk = torch.topk(W.abs().flatten(), k=int(W.numel() * sparsity), largest=False)
                # topk will have .indices and .values
                mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
                mask[topk.indices] = 0
                mask = mask.reshape(W.shape)
                del topk


        if iters == 0:
            Z = (W) * mask
            out = (Z.T / norm)
            print((out == 0).sum().item() / out.numel())

            self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            return

        rho = 1
        XY = XX.matmul(W)
#        XX = XX + torch.eye(XX.shape[1], device=XX.device)*rho  
        XX[diag,diag] += rho
        torch.cuda.empty_cache()
        XXinv = torch.inverse(XX)
        self.XX = None
        del XX
        U = torch.zeros_like(W)

        for itt in range(iters): # 20
            if iterative_prune > 0 and itt < iterative_prune:
                if prune_n != 0:
                    sparsity = prune_n / prune_m # fixed sparsity for n:m(n count-->0 /m)
                    cur_sparsity = sparsity - sparsity * (1 - (itt + 1) / iterative_prune) ** 3
                    # WT = (W+U).T.reshape((W.shape[1]*W.shape[0]//4, 4)).abs() #[4096//4, 4]
                    # # WT = (W+U).T.reshape((W.shape[1]*W.shape[0]//m, m)).abs()
                    # mask = torch.zeros(WT.shape, dtype=torch.bool)
                    # sort_inds = WT.sort(dim=1)[1]
                    # mask[torch.arange(WT.shape[0]), sort_inds[:,2]] = 1
                    # mask[torch.arange(WT.shape[0]), sort_inds[:,3]] = 1
                    # mask = mask.reshape(W.T.shape).T
                    WT = (W+U).T.reshape(-1, prune_m).abs()
                    mask = torch.zeros_like(WT, dtype=torch.bool)
                    sort_inds = WT.argsort(dim=1)
                    mask.scatter_(1, sort_inds[:, -(prune_m-prune_n):], 1)
                    mask = mask.reshape(W.T.shape).T

                    Z2 = (W+U).abs()
                    Z2[mask] = torch.inf
                    thres = Z2.flatten().kthvalue(int(W.numel() * cur_sparsity)+1)[0]
                    mask = (Z2 >= thres)
                    del thres

                else:
                    cur_sparsity = sparsity - sparsity * (1 - (itt + 1) / iterative_prune) ** 3
                    if per_out:
                        thres = (W+U).abs().sort(dim=0)[0][int(W.shape[0] * cur_sparsity)]
                        mask = ((W+U).abs() >= thres.unsqueeze(0))
                        del thres
                    else:
                        topk = torch.topk((W+U).abs().flatten(), k=int(W.numel() * sparsity), largest=False)
                        # topk will have .indices and .values
                        mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
                        mask[topk.indices] = 0
                        mask = mask.reshape(W.shape)
                        del topk

            Z = (W + U) * mask


            U = U + (W - Z)

            W = XXinv.matmul(XY + rho*(Z-U))

        Z = (W + U) * mask

        out = (Z.T / norm)
        # if self.args.d2_admm:
        #     # out = (Z.T / (norm + self.r1 * (norm_y ** (1/2)) * (norm_x** (1/2)) - self.r2 * (norm_x ** (1))))
        #     out = out = (Z.T / (beta*norm))
        print((out == 0).sum().item() / out.numel())

        self.layer.weight.data = out.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
    

    def free(self):
        self.XX = None
        torch.cuda.empty_cache()

