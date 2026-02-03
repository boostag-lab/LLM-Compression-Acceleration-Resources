
import os
import torch
from lm_eval.utils import make_table

from cfg import get_args
from data import get_dataloader
from model import get_model
from prune import D2Prune, Pruner
from utils import eval_ppl, eval_zero_shot

import torch
# print("Torch CUDA:", torch.version.cuda)
# print("Torch cuDNN:", torch.backends.cudnn.versio
#-----------------------------------loading args from parameters yaml file----------------------------------------------#
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

cfg_path = './cfg/model.yaml'
args = get_args(cfg_path)

#-----------------------------------loading model and tokenizer---------------------------------------------------------
if args.free:
    model, tokenizer = get_model(args.model, device_type="cpu", seq_len=args.seqlen) # cpu loading
else:
    model, tokenizer = get_model(args.model, device_type="auto", seq_len=args.seqlen) # gpu loading
model.eval()
args.seq_len = model.seq_len
model_name = args.model.split("/")[-1]
args.model_name = model_name

def main(demo=False):
    if args.sparsity_ratio != 0:
        # loading calibration dataloader
        train_loader = get_dataloader(args, tokenizer, model.seq_len, args.cali_dataset, eval_mode=False)

        args.logger.info("pruning starts")
        if args.prune_method == 'd2prune':
            pruner = D2Prune(args, model).pruner
        else:
            if args.prune_method == 'pruner-zero':
                from prune.pruner_zero import get_layer_gradient, GPTree
                args.gradients_l2 = get_layer_gradient(args, model, train_loader, args.device, data_cache_dir='./cache')
                args.engine = GPTree.load_tree('./prune/pruner_zero/best_tree.json')
            pruner = Pruner(args, model).pruner
        pruner.prune_llm(train_loader)

        # Save the pruned model
        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            args.logger.info(f"save model to {args.save_model}")

    args.save_filepath = os.path.join(args.output_dir, f"log_{args.prune_method}.txt")

    # loading eval dataloader
    eval_loader = get_dataloader(args, tokenizer, model.seq_len, args.eval_data_path, eval_mode=True) # Tuple[Tensor,Tensor]
    test_loader = torch.stack([loader[0] for loader in eval_loader]) # [n,1,2048]->Tensor

    # ppl test
    if not args.free:
        ppl_test = eval_ppl(args, model, test_loader)
    ## device offloading
    else:
        ppl_test = eval_ppl(args, model, test_loader, is_split=True)
    ## ppl_test = eval_ppl(args, model, test_loader, is_split=True)

    # zero-shot acc test
    if args.eval_zero_shot:
        task_list = None
        if demo:
            task_list = ['boolq']
        results = eval_zero_shot(args, model, tokenizer, task_list=task_list)
        args.logger.info("\n" + make_table(results))
        with open(args.save_filepath, "w") as f:
            if ppl_test:
                print("********************************")
                print("method\tsparsity\tppl_test", file=f, flush=True)
                print(f"{args.prune_method}\t{args.sparsity_ratio}\t{ppl_test:.4f}", file=f,
                      flush=True)
            print("********************************")
            print("zero_shot_results", file=f, flush=True)
            print(make_table(results), file=f, flush=True)
        args.logger.info(f"save filepath:{args.save_filepath}")


if __name__ == "__main__":
    main()