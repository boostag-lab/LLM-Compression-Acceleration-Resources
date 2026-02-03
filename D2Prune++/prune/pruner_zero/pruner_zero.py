import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import random
from loguru import logger
import gc
import time
from sklearn.cluster import KMeans
import math
import sys

DEBUG = False
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaTokenizer)

try:
    from transformers import AdamW # trasformers<4.45.0
except:
    from torch.optim import AdamW # trasformers>=4.45.0
from .gradient_computation import GradientComputation, ActivationComputation
import torch
import json
import math
import torch
import os
from loguru import logger
import gc
MIN_DEPTH = 2  # minimal initial random tree depth
MAX_DEPTH = 4  # maximal initial random tree depth
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class PrunerZero:
    """
    This class wraps a GPT layer for specific operations.
    Pruner-Zero:https://github.com/pprp/Pruner-Zero
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

    def fasterprune(self, sparsity, prune_n, prune_m, gradients, engine):
        '''
        :param gradients: layer gradients
        :param engine: metric engine (W,X,G)
        :return:
        '''
        # W_metric = engine.forward(
        #     torch.abs(self.layer.weight.data).to(dtype=torch.float32),
        #     gradients.to(device=self.dev, dtype=torch.float32),
        #     self.scaler_row.reshape((1, -1)).to(device=self.dev, dtype=torch.float32),
        # )
        W_metric = engine.forward(
            torch.abs(self.layer.weight.data).to(dtype=torch.float32),
            gradients.to(device=self.dev, dtype=torch.float32),
            # self.scaler_row.reshape((1, -1)).to(device=self.dev, dtype=torch.float32),
        )
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
        gradients = None
        torch.cuda.empty_cache()

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


def get_layer_gradient(args, model, train_loader, device, task='gradient', scale=100, data_cache_dir='./dataset/cache'):
    '''
    model device --> better gpu, otherwise slowly train
    '''
    model_name = args.model_name
    gradient_cache =  f'{data_cache_dir}/cali_{args.cali_dataset}_{model_name}_gradients_nsamples{args.nsamples}.cache'
    if "qwen" in model_name.lower() or "mistral" in model_name.lower():
        gradient_cache =  f'{data_cache_dir}/cali_wikitext2_{model_name}_gradients_nsamples{args.nsamples}.cache'
    if os.path.exists(gradient_cache):
        logger.info(f"load gradient from {gradient_cache}")
        gradients_l2 = torch.load(gradient_cache, map_location=torch.device('cpu'))
        return gradients_l2
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    # model, optimizer = accelerator.prepare(model, optimizer)
    optimizer.zero_grad()
    if task == 'gradient':
        computer = GradientComputation(model, scale)
    elif task == 'activation':
        computer = ActivationComputation(model)
    else:
        raise ValueError(f'task {task} not supported')
    if "70b" not in model_name:
        nsample = 0
        model.train()
        for i, input_ids in enumerate(train_loader):
            if i % 50 == 0:
                print(f"sample {i} training for gradient")
            nsample += 1
            try:
                input_ids = input_ids.to(device)
            except:
                input_ids = input_ids[0].to(device)
            labels = input_ids.clone()
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            # outputs = model(input_ids=input_ids, labels=input_ids.clone())
            loss = outputs.loss
            loss.backward()
            # accelerator.backward(loss)
            optimizer.step()
            # optimizer.zero_grad()
            if task == 'gradient':
                computer.update_gradient(model, nsample)
            elif task == 'activation':
                computer.update_activation()
            optimizer.zero_grad()
            del input_ids
            # del labels
            del outputs
            del loss
            gc.collect()
            torch.cuda.empty_cache()
        print('Done')
    gradients_l2 = computer.gradients_l2
    for name in gradients_l2:
        grad_sqrt = torch.sqrt(gradients_l2[name])
        gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)
    try:
        torch.save(gradients_l2, gradient_cache)
        print(f"save gradients to {gradient_cache}, now exit")
        sys.exit(0) # exit after saving gradients
    except:
        pass
    torch.cuda.empty_cache()
    
    return gradients_l2