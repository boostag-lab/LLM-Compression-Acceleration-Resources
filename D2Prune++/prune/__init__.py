from .d2prune_utils import D2SparseGPT, D2Wanda
import transformers 
if transformers.__version__ >= '4.51.0':
    from .prune_llama_v2 import D2Prune_LLAMA, Prune_LLAMA
else:
    from .prune_llama import D2Prune_LLAMA, Prune_LLAMA
from .prune_opt import D2Prune_OPT, Prune_OPT
from .prune_qwen import D2Prune_QWEN, Prune_QWEN
from .prune_mistral import D2Prune_MISTRAL, Prune_MISTRAL


class D2Prune:
    def __init__(self, args, model):
        self.args = args
        if 'llama' in self.args.model.lower():
            self.pruner = D2Prune_LLAMA(args, model)
        elif 'opt' in self.args.model.lower():
            self.pruner = D2Prune_OPT(args, model)
        elif 'qwen' in self.args.model.lower():
            self.pruner = D2Prune_QWEN(args, model)
        elif 'mistral' in self.args.model.lower():
            self.pruner = D2Prune_MISTRAL(args, model)
        else:
            raise ValueError(f'Unsupported model {self.args.model.lower()}, please check your model path')

class Pruner:
    def __init__(self, args, model):
        '''
        this class is for SparseGPT/Wanda/Pruner-Zero
        :param args:
        :param model:
        '''
        self.args = args
        if 'llama' in self.args.model.lower():
            self.pruner = Prune_LLAMA(args, model)
        elif 'opt' in self.args.model.lower():
            self.pruner = Prune_OPT(args, model)
        elif 'qwen' in self.args.model.lower():
            self.pruner = Prune_QWEN(args, model)
        elif 'mistral' in self.args.model.lower():
            self.pruner = Prune_MISTRAL(args, model)
        else:
            raise ValueError(f'Unsupported model {self.args.model.lower()}, please check your model path')

