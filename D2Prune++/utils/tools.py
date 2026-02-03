import os
import random
import sys
import time
from functools import wraps

import numpy as np
import toml
import torch
import torch.nn as nn
import yaml
from loguru import logger


## yaml
def read_yaml_to_dict(yaml_path: str):
    """yaml file to dict"""
    with open(yaml_path, 'r') as file:
        dict_value = yaml.safe_load(file)
        return dict_value

def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict save to yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

## toml
def read_toml_to_dict(toml_path: str):
    """toml file to dict"""
    with open(toml_path, 'r') as file:
        dict_value = toml.load(file)
        return dict_value

def save_toml_to_dict(dict_value: dict, save_path: str):
    """dict save to toml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(toml.dumps(dict_value))


def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logger(log_name, save_dir):
    filename = '%s.log' % log_name
    save_file = os.path.join(save_dir, filename)
    if os.path.exists(save_file):
        with open(save_file, "w") as log_file:
            log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB", format="{time} {level} {message}", level="INFO")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info('This is the %s log' % log_name)
    return logger


def timeit(func):
    """
    running time evaluation
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration:.6f} seconds to run.")
        return result
    return wrapper


def find_layers(module, layers=[nn.Linear], name=''):
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

    layers = model.model.decoder.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            # count += (W==0).sum().item()
            count += (W == 0).sum().cpu().item()
            total_params += W.numel()
            # sub_count += (W == 0).sum().item()
            sub_count += (W==0).sum().cpu().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params