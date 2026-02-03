from model.llama import LLAMA
from model.opt import OPT
from model.qwen import Qwen
from model.mistral import Mistral
from model.deit import DeiT
from utils import timeit


@timeit
def get_model(model_path: str, device_type: str, seq_len=None):
    '''
    :param model_path: -> model file path or huggingface model name
    :param device_type: 'cpu', 'cuda:0', 'auto'
    :param seq_len: 2048 if save memory, else None (default)
    :return: model and tokenizer
    '''
    if "llama" in model_path.lower():
        llm = LLAMA(model_path, device_type)
        model = llm.load_model(seq_len)
        tokenizer = llm.load_tokenizer()
        return model, tokenizer
    elif "opt" in model_path:
        llm = OPT(model_path, device_type)
        model = llm.load_model(seq_len)
        tokenizer = llm.load_tokenizer()
        return model, tokenizer
    elif "qwen" in model_path.lower():
        llm = Qwen(model_path, device_type)
        model = llm.load_model(seq_len=seq_len if seq_len else 4096)
        tokenizer = llm.load_tokenizer()
        return model, tokenizer
    elif "mistral" in model_path.lower():
        llm = Mistral(model_path, device_type)
        model = llm.load_model(seq_len=seq_len if seq_len else 4096)
        tokenizer = llm.load_tokenizer()
        return model, tokenizer

    else:
        raise ValueError(f'Unknown model {model_path}')