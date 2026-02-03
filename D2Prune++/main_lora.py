## run lora train llm and evaluate llm
## train_dataset:c4
## eval_dataset:wikitext2
from lora_ft import get_lora_args, LoRATrainer
from utils import setup_logger
import os
from lib.prune_llama import check_sparsity as check_sparsity_llama
from lib.prune_opt import check_sparsity as check_sparsity_opt
from lib.eval import eval_ppl, eval_zero_shot
from cfg.args import Prune_Args
import torch
from datasets import load_dataset
import torch.nn as nn
from lora_ft import eval_zero_shot, get_wikitext2, eval_ppl_wikitext
from lm_eval.utils import make_table

#--------------------------------------------------lora training args---------------------------------------------------
model_args, data_args, training_args = get_lora_args()
if model_args.model_name_or_path:
    # model_args.logger = setup_logger("lora", os.path.join('lora_out', model_args.model_name_or_path.split('/')[-1]))
    model_args.logger = setup_logger("lora", os.path.join('lora_out', training_args.output_dir.split('/')[-1]))
    # model_args.logger = setup_logger("lora_acc", os.path.join('lora_out', training_args.output_dir.split('/')[-1]))


def main():
    # --------------------------------------------------Lora Trainer---------------------------------------------------------

    trainer = LoRATrainer(model_args, data_args, training_args)
    logger = trainer.logger
    model, tokenizer = trainer.load_model()
    model.eval()
    _, testloader = get_wikitext2(model_name=model_args.model_name_or_path, nsamples=128, seed=0, seqlen=model.seqlen,
                                  tokenizer=tokenizer,
                                  local_path='../cache/data/wikitext', cache_dir='./dataset/cache')

    # before lora
    logger.info(f"model strcture:{model}")
    ppl_before_lora = eval_ppl_wikitext(model, testloader, bs=1, device=training_args.device)
    logger.info(f"{model_args.model_name_or_path.split('/')[-1]} eval ppl before lora:{ppl_before_lora}")
    del model
    torch.cuda.empty_cache()

    trainer.train()

    print("*" * 30)
    if "opt" in model_args.model_name_or_path.lower():
        sparsity_ratio = check_sparsity_opt(trainer.model)
    elif "llama" in model_args.model_name_or_path.lower():
        sparsity_ratio = check_sparsity_llama(trainer.model)
    else:
        raise NotImplementedError
    model = trainer.model
    del trainer
    torch.cuda.empty_cache()
    model.eval()
    # after lora
    # model = PeftModel.from_pretrained(model, training_args.output_dir, torch_dtype=torch.float16)
    logger.info(f"lora model strcture:{model}")
    ppl_after_lora = eval_ppl_wikitext(model, testloader,bs=1, device="cuda")
    logger.info(f"ppl_after_lora eval ppl:{ppl_after_lora}")

    # eval-zero-shot
    # tasks = ["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
    model_args.device = training_args.device
    if model_args.eval_zero_shot:
        # tasks = ['boolq']
        # results = eval_zero_shot(model_args, model, tokenizer, task_list=tasks)
        task_list = ["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
        results = eval_zero_shot(model_args, model, tokenizer, task_list, batch_size=4)
        logger.info("\n" + make_table(results))




if __name__ == "__main__":
    main()




