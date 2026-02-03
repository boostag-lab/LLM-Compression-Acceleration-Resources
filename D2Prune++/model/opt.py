"""
opt-125m
opt-350m
opt-2.7b
opt-6.7b
opt-13b
"""

from typing import List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OPTForCausalLM

from .base_model import LLM

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer


class OPT(LLM):
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
            model = OPTForCausalLM.from_pretrained(self.model_path, torch_dtype='auto',
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
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                  cache_dir=self.cache_dir,
                                                  use_fast=False, legacy=False)
        return tokenizer


    def load_layers(self, model, model_type) -> List[torch.nn.Module]:
        if model_type == OPT_MODEL:
            return [layer for layer in model.model.decoder.layers]
        else:
            raise ValueError(f'Unknown model type {model_type}')

    def load_embedding(self, model, model_type) -> List[torch.nn.Module]:
        if model_type == OPT_MODEL:
            return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
        else:
            raise ValueError(f'Unknown model type {model_type}')




