from .data_load import PruneDataset

def get_dataloader(args, tokenizer, seq_len, data_path, eval_mode):
    if eval_mode:
        prune_dataset = PruneDataset(args, args.eval_dataset, tokenizer, seq_len, data_path, eval_mode=True)
    else:
        prune_dataset = PruneDataset(args, args.cali_dataset, tokenizer, seq_len, data_path, eval_mode=False)
    return prune_dataset.dataloader