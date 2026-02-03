"""
opt family
llama family
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from abc import ABCMeta, abstractmethod
import numpy as np
import random

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

class LLM(metaclass=ABCMeta):
    """
    set seed
    load model
    load tokenizer
    load layers
    load embedding
    """
    def __init__(self, model_path, cache_dir="llm_weights"):
        """
        model_type-->choose one of ["auto", "cpu", "offload", "cuda:0]
        """
        self.model_path = model_path
        self.cache_dir = cache_dir

    def setup_seed(self, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @abstractmethod
    def load_model(self, model_type):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def load_layers(self, model, model_type):
        pass

    @abstractmethod
    def load_embedding(self, model, model_type):
        pass