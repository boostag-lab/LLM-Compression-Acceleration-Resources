
from loguru import logger
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from itertools import chain
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Trainer, default_data_collator, is_torch_tpu_available
import math

os.environ["WANDB_DISABLED"] = "true"
class LoRATrainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.logger = model_args.logger
        # lora weight output dir
        self.output_dir = training_args.output_dir
        # dataset name
        self.dataset_name = data_args.dataset_name
        # cache data dir
        self.data_cache_dir = data_args.data_cache_dir
        # model path
        self.model_name_or_path = model_args.model_name_or_path
        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()
        # Log on each process the small summary:
        self.logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        self.logger.info(f"Training/evaluation parameters {training_args}\n model parameters {model_args}\n data parameters {data_args}")


    def load_datasets(self):
        if self.data_args.local_data_path: #
            self.logger.info('load data from the local directory')
            self.raw_datasets = load_dataset(self.data_args.local_data_path)
        else: #
            self.logger.info('load data from huggingface, make sure that you have opened vpn and download the dataset from huggingface')
            if self.dataset_name == "c4":
                self.raw_datasets = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'})
            elif self.dataset_name in ["wikitext2", "wikitext"]:
                self.raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
            else:
                raise NotImplementedError

        # self.raw_datasets['train'] = self.raw_datasets['train'].select(
        #     range(min(2000, len(self.raw_datasets['train']))))
        # self.raw_datasets['validation'] = self.raw_datasets['validation'].select(
        #     range(min(2000, len(self.raw_datasets['validation']))))
        return self.raw_datasets

    def process_datasets(self):
        if self.training_args.do_train:
            column_names = list(self.raw_datasets["train"].features)
        else:
            column_names = list(self.raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = self.tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with self.training_args.main_process_first(desc="dataset map tokenization"):
            if not self.data_args.streaming:
                tokenized_datasets = self.raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = self.raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
        if self.data_args.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                self.logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if self.data_args.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.data_args.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.data_args.block_size, self.tokenizer.model_max_length)

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        with self.training_args.main_process_first(desc="grouping texts together"):
            if not self.data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
        return lm_datasets


    def load_model(self):
        self.logger.info(f"loading model and tokenizer from {self.model_name_or_path}")
        # training_args.device
        if self.training_args.device != "cpu" and torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16,
                                                         cache_dir=self.model_args.cache_dir, low_cpu_mem_usage=True,
                                                         device_map=self.training_args.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16,
                                                         cache_dir=self.model_args.cache_dir, low_cpu_mem_usage=True,
                                                         device_map="auto")
        model.seqlen = model.config.max_position_embeddings
        tokenizer =  AutoTokenizer.from_pretrained(self.model_name_or_path,cache_dir=self.model_args.cache_dir,
                                                   use_fast=False, legacy=False)
        if self.model_args.config_name and "decapoda-research" in self.model_args.config_name: #llama-7b, need vpn if first download
            tokenizer = AutoTokenizer.from_pretrained(
                "lmsys/vicuna-13b-delta-v0",
                cache_dir=self.model_args.cache_dir,
                padding_side="right",
                use_fast=True,
            )
        return model, tokenizer

    # training args
    def initialize_training_args(self):
        batch_size = 128
        self.training_args.gradient_accumulation_steps = batch_size // self.training_args.per_device_train_batch_size
        self.training_args.warmup_steps = 5
        self.training_args.fp16 = True
        self.training_args.logging_steps = 10
        self.training_args.optim = "adamw_torch"
        self.training_args.save_strategy = "steps"
        self.training_args.eval_steps = 10
        self.training_args.save_steps = 50
        self.training_args.save_total_limit = 15
        self.training_args.group_by_length = False


    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def train(self):
        # checkpoint, which we can resume from
        last_checkpoint = None
        if os.path.isdir(self.output_dir) and self.training_args.do_train and not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.output_dir)
            if last_checkpoint is None and len(os.listdir(self.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(self.training_args.seed)

        self.model, self.tokenizer = self.load_model()

        # int 8 training
        self.model = prepare_model_for_int8_training(self.model)

        # lora config
        config = LoraConfig(
            r=self.model_args.lora_r,
            lora_alpha=self.model_args.lora_alpha,
            # target_modules=["q_proj", "v_proj"],
            target_modules=self.model_args.target_modules,
            lora_dropout=self.model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # lora model
        self.model = get_peft_model(self.model, config)
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        model_name = self.tokenizer.name_or_path.split("/")[-1]

        train_cache_dataloader = f'{self.data_cache_dir}/train_{self.dataset_name}_{model_name}_samples{self.data_args.max_train_samples}_lora.cache'
        eval_cache_dataloader = f'{self.data_cache_dir}/eval_{self.dataset_name}_{model_name}_samples{self.data_args.max_eval_samples}_lora.cache'
        if os.path.exists(train_cache_dataloader) or os.path.exists(eval_cache_dataloader):
            if os.path.exists(train_cache_dataloader):
                self.logger.info(f"load train processed data from {train_cache_dataloader}")
                self.train_dataset = torch.load(train_cache_dataloader)
            if os.path.exists(eval_cache_dataloader):
                self.logger.info(f"load eval processed data from {eval_cache_dataloader}")
                self.eval_dataset = torch.load(eval_cache_dataloader)
        else:
            self.logger.info(f"start processing data")
            # Get the datasets
            self.raw_datasets = self.load_datasets()
            # processed datasets
            self.lm_datasets = self.process_datasets()

            # train datasests(max_train_examples), full datasets or select max_train_samples
            if self.training_args.do_train:
                if "train" not in self.lm_datasets:
                    raise ValueError("--do_train requires a train dataset")
                self.train_dataset = self.lm_datasets["train"]
                if self.data_args.max_train_samples is not None:
                    max_train_samples = min(len(self.train_dataset), self.data_args.max_train_samples)
                    self.train_dataset = self.train_dataset.select(range(max_train_samples))
                if self.data_args.data_cache_dir:
                    torch.save(self.train_dataset, train_cache_dataloader)
                    self.logger.info(f"saving train processed data to {train_cache_dataloader}")
            # validation datasets(max_eval_samples)
            if self.training_args.do_eval:
                if "validation" not in self.lm_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                self.eval_dataset = self.lm_datasets["validation"]
                if self.data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(self.eval_dataset), self.data_args.max_eval_samples)
                    self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))
                if self.data_args.data_cache_dir:
                    torch.save(self.eval_dataset, eval_cache_dataloader)
                    self.logger.info(f"saving eval processed data to {eval_cache_dataloader}")
        self.logger.info(f"loading processed data finished")

        # Initialize our Trainer
        self.initialize_training_args()

        def accuracy(predictions, references):
            if len(predictions) != len(references):
                raise ValueError("Predictions and references should have the same length.")
            correct_count = sum([1 for pred, ref in zip(predictions, references) if pred == ref])
            return correct_count / len(predictions)

        # # metric
        # metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            acc = accuracy(predictions=preds, references=labels)
            return acc

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if self.training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            if self.training_args.do_eval and not is_torch_tpu_available()
            else None,
        )
        self.model.config.use_cache = False

        # Training

        if self.training_args.do_train:
            self.logger.info("lora start training")
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # save model
        self.model.save_pretrained(self.training_args.output_dir)
        torch.save(trainer.model.state_dict(), f"{self.training_args.output_dir}/adapter_model.bin")

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        if self.training_args.do_eval:
            self.evaluate(trainer)
        kwargs = {"finetuned_from": self.model_args.model_name_or_path, "tasks": "text-generation"}
        if self.data_args.dataset_name is not None:
            kwargs["dataset_tags"] = self.data_args.dataset_name
            if self.data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = self.data_args.dataset_config_name
                kwargs["dataset"] = f"{self.data_args.dataset_name} {self.data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = self.data_args.dataset_name

        if self.training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            self.logger.info("training finished")

        #   # Evaluation
    def evaluate(self, trainer):
        self.logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)