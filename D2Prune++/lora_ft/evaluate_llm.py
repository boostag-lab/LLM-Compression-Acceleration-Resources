from datasets import load_dataset
import os
import os
import random
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from typing import List

from utils import cleanup_memory, distribute_model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    try:
        layers = model.model.layers
    except:
        layers = model.model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            cur_zeros = (W==0).sum().item()
            cur_total = W.numel()

            count += cur_zeros
            total_params += cur_total

            print(f"layer {i} name {name} {W.shape} sparsity {float(cur_zeros)/cur_total}")

    print(f"total number of params {total_params}")
    model.config.use_cache = use_cache
    return float(count)/total_params

def get_wikitext2(model_name, nsamples, seed, seqlen, tokenizer,
                  local_path='../wanda/data/wikitext',
                             cache_dir='./data/cache'):
    model_name = model_name.lower()
    cache_dataloader = f'{cache_dir}/eval_wikitext2_{model_name}_seqlen{seqlen}_nsamples{nsamples}.cache'
    if os.path.exists(cache_dataloader):
        print(f"load eval data from {cache_dataloader}")
        train_loader, testenc = torch.load(cache_dataloader)
        return train_loader, testenc
    traindata = load_dataset(local_path, split='train')
    testdata = load_dataset(local_path, split='test')
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    try:
        torch.save([trainloader,testenc], cache_dataloader)
    except:
        pass
    return trainloader, testenc

@torch.no_grad()
def eval_ppl_wikitext(model, testenc,bs=1, device=None):
    model.seqlen = model.config.max_position_embeddings
    # Get input IDs
    testenc = testenc.input_ids
    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    return ppl.item()

def eval_zero_shot(args, model, tokenizer, task_list: List[str] = None, batch_size = 8):
    cleanup_memory()

    if args.distribute: # 70b
        distribute_model(model)
    else:
        model.to(args.device)

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
    mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in
                results['results'].items()}
    mean_std_val = round(sum(std_vals.values()) / len(std_vals.values()), 4)
    metric_vals['acc_avg'] = mean_acc_val
    results['results']['AVERAGE'] = {
        "acc,none": mean_acc_val,
        "acc_stderr,none": mean_std_val
    }
    return results
