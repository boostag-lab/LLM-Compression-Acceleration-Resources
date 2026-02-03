import time
from typing import List

import lm_eval
import torch
import torch.nn as nn
from lm_eval.models.huggingface import HFLM
from loguru import logger
from tqdm import tqdm
from .memory import cleanup_memory, distribute_model
from .tools import timeit


def eval_ppl(args, model, test_loader: torch.Tensor, is_split=False):
    with torch.no_grad():
        if '70b' in args.model.lower() or is_split:
            ppl = eval_ppl_split(args, model, test_loader)
        else:
            ppl = eval_ppl_no_split(args, model, test_loader)
    return ppl

@timeit
def eval_ppl_no_split(args, model, test_loader):
    '''
    suitable for models <70b, cpu/single gpu/
    '''
    nlls = []
    nsamples = len(test_loader)
    logger.info(f"eval data total samples:{nsamples}")
    if model.device == 'cpu':
        model = model.to(args.device)
    start = time.time()
    for i, inputs in tqdm(enumerate(test_loader)):
        if i % 50 == 0:
            args.logger.info(f"sample {i}")
        inputs = inputs.reshape(1, model.seq_len).to(args.device)
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:] # window is 1?
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()

    return ppl.item()

@timeit
def eval_ppl_split(args, model, test_loader):
    if 'opt' in args.model.lower():
        ppl = opt_eval(args, model, test_loader)
        return ppl
    elif 'llama' in args.model.lower():
        import transformers
        if transformers.__version__ >= '4.51.0':
            ppl = llama_eval_v2(args, model, test_loader)
        else:
            ppl = llama_eval(args, model, test_loader)
        return ppl
    elif 'mistral' in args.model.lower():
        ppl = mistral_eval(args, model, test_loader)
        return ppl
    
    elif 'qwen' in args.model.lower():
        ppl = qwen_eval(args, model, test_loader)
        return ppl
    else:
        raise ValueError(f'Unknown model {args.model}')

@torch.no_grad()
def opt_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    device = model.model.decoder.embed_tokens.weight.device
    if device.type == 'cpu':
        model.model.decoder.embed_tokens.to(args.device)
        model.model.decoder.embed_positions.to(args.device)
        model.model.decoder.final_layer_norm.to(args.device)
        device = args.device
    else:
        device = device.index
    print(f"device:{device}")
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.decoder.embed_tokens.to("cpu")
    model.model.decoder.embed_positions.to("cpu")
    model.model.decoder.final_layer_norm.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(args.device)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_eval_v2(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
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
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()


@torch.no_grad()
def mistral_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
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
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def qwen_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
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
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()


@timeit
def eval_zero_shot(args, model, tokenizer, task_list: List[str] = None, batch_size = 8):
    cleanup_memory()

    if args.distribute: # 70b
        distribute_model(model)
    else:
        model.to(args.device)
        
    if '66b' in args.model:
        batch_size= batch_size // 2

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    task_manager = lm_eval.tasks.TaskManager()
    if not task_list:
        task_list = args.tasks
    tasks = task_manager.match_tasks(task_list)
    start = time.time()
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=batch_size)
    args.logger.info(f"total eval zero shot run time:{(time.time() - start):.2f} seconds")
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in
                   results['results'].items()}
    metric_vals1 = {
        task: round(max(result.get('acc,none', 0), result.get('acc_norm,none', 0)), 4)
        for task, result in results['results'].items()
    }
    mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    mean_acc_val1 = round(sum(metric_vals1.values()) / len(metric_vals1.values()), 4)
    std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in
                results['results'].items()}
    mean_std_val = round(sum(std_vals.values()) / len(std_vals.values()), 4)
    metric_vals['acc_avg'] = mean_acc_val
    results['results']['AVERAGE'] = {
        "acc,none": mean_acc_val,
        "acc_stderr,none": mean_std_val
    }
    results['results']['AVERAGE1'] = {
        "acc,none": mean_acc_val1,
        "acc_stderr,none": mean_std_val
    }
    return results




















