import torch.nn as nn
import torch

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class Wanda:
    """
    This class wraps a GPT layer for specific operations.
    ## Wanda: https://github.com/locuslab/wanda
    """
    def __init__(self, args, layer, layer_id=0, layer_name="none"):
        self.args = args
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out): # inp-->[2048,768]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0) # [1,2048,768]
        tmp = inp.shape[0] # 1
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) # [2048,768]
            inp = inp.t() # [768,2048]

        self.scaler_row *= self.nsamples / (self.nsamples+tmp) # [768] * 0/(0+1)
        self.nsamples += tmp # 1

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples # [768] + ||inp||^2/1

    def fasterprune(self, sparsity, prune_n=0, prune_m=0):
        # org-->scaling to sqrt, Cauchy-Schwarz inequality
        W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
        W_mask = (torch.zeros_like(W_metric) == 1)
        if prune_n != 0:
            self.args.logger.info(f"pruning N:M--> {prune_n}:{prune_m}")
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)
        self.layer.weight.data[W_mask] = 0
        W_metric = None
        W_mask = None
        torch.cuda.empty_cache()

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()