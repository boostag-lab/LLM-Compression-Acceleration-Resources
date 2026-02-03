from model import get_model

if __name__ == '__main__':
    model_path = '../../../cache/llm_weights/llama-2-70b'
    model, tokenizer = get_model(model_path, device_type="auto")
    print(model.seq_len)
    print(tokenizer.name_or_path.split("/")[-1])