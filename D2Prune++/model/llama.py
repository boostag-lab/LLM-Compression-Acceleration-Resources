"""
llama-7b
llama-13b
llama-30b
llama-2-7b
llama-2-13b
llama-2-70b
"""
from typing import List

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

from .base_model import LLM

LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


class LLAMA(LLM):
    """
    Initialize LLAMA model and tokenizer.
    Args:
        model_name (str): Model name to load.
        model_path (str): Path to the model, such as 'llm_weights/models--facebook--opt-125m/snapshots/opt-125m'
        model_type (str): Device to load the model on.['cpu', 'cuda', 'mps', 'auto','cuda:1]
    """
    def __init__(self, model_path, device_type, model_name=None):
        super().__init__(model_path)
        self.device_type = device_type
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]


    def load_model(self, seq_len=None):
        if seq_len: # saving memory
            def skip(*args, **kwargs):
                pass
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip
            model = LlamaForCausalLM.from_pretrained(self.model_path, torch_dtype='auto',
                                                     low_cpu_mem_usage=True, device_map=self.device_type)
            model.seq_len = seq_len
            return model

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_type
        )
        model.seq_len = model.config.max_position_embeddings
        return model  # seq_len = model.config.max_position_embeddings-->4096

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                  cache_dir=self.cache_dir,
                                                  use_fast=False, legacy=False)
        return tokenizer

    def load_layers(self, model, model_type) -> List[torch.nn.Module]:
        if model_type == LLAMA_MODEL:
            return [layer for layer in model.model.layers]
        else:
            raise ValueError(f'Unknown model type {model_type}')

    def load_embedding(self, model, model_type) -> List[torch.nn.Module]:
        if model_type == LLAMA_MODEL:
            return [model.model.embed_tokens]
        else:
            raise ValueError(f'Unknown model type {model_type}')




