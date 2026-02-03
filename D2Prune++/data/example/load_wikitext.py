
from data import PruneDataset

if __name__ == '__main__':
    # Load the dataset
    from cfg import get_args
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def get_model(model_path, model_type="auto"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=model_type,

        )
        model.seq_len = model.config.max_position_embeddings
        return model
    model_path = '../../../cache/llm_weights/models--facebook--opt-125m/snapshots/opt-125m'

    model = get_model(model_path, model_type="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = get_args('../../cfg/model.yaml')
    eval_prune_dataset = PruneDataset(args, args.eval_dataset,tokenizer, model.seq_len,
                                      '../../../cache/data/wikitext',
                                      eval_mode=True)
    dataloader = eval_prune_dataset.dataloader
    print(len(dataloader))
    print(dataloader[0])
    print(dataloader[0][0].shape)


