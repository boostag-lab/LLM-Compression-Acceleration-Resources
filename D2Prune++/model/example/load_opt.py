from model import get_model

if __name__ == '__main__':
    model_path = '../../../cache/llm_weights/models--facebook--opt-125m/snapshots/opt-125m'
    model, tokenizer = get_model(model_path, device_type="cpu")
    print(model.seq_len)
    print(tokenizer.name_or_path.split("/")[-1])