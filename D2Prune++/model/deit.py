from transformers import DeiTForImageClassification, DeiTConfig, \
    AutoFeatureExtractor,AutoImageProcessor,AutoModelForImageClassification


class DeiT:
    def __init__(self, args, model_path, device_type=None, model_name=None):
        self.args = args
        self.model_path = model_path
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]
        self.device_type = device_type

    def load_model(self):
        self.model = AutoModelForImageClassification.from_pretrained(self.model_path,
                                                                     device_map=self.device_type,
                                                                     # torch_dtype=torch.float16,
                                                                     low_cpu_mem_usage=True)
        return self.model

    def load_processer(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        return self.processor