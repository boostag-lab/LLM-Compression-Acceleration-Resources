"""
Qwen3-8B
Qwen3-14B
"""
from typing import List

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

from .base_model import LLM


class Mistral(LLM):
    """
    Initialize LLAMA model and tokenizer.
    Args:
        model_name (str): Model name to load.
        model_path (str): Path to the model, such as '/llm_weights/Qwen3-8B'
        model_type (str): Device to load the model on.['cpu', 'cuda', 'mps', 'auto','cuda:1]
    """
    def __init__(self, model_path, device_type, model_name=None):
        super().__init__(model_path)
        self.device_type = device_type
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]


    def load_model(self, seq_len=4096):
        '''
        model.config.max_position_embeddings-->32768
        '''
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_type
        )
        model.seq_len = model.config.max_position_embeddings if model.config.max_position_embeddings <=4096 else seq_len
        return model  # seq_len = model.config.max_position_embeddings-->4096

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                  cache_dir=self.cache_dir,
                                                  use_fast=False, legacy=False)
        return tokenizer

    def load_layers(self, model, model_type) -> List[torch.nn.Module]:

        return [layer for layer in model.model.layers]


    def load_embedding(self, model, model_type) -> List[torch.nn.Module]:

        return [model.model.embed_tokens]





