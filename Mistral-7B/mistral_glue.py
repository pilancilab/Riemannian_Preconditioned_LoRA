import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
import numpy as np
import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForSequenceClassification, TrainingArguments
import mistral
from mistral.trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default='scaled_adamw', type=str,
                        choices=['adamw', 'scaled_adamw', 'sgd', 'scaled_gd'])
cmd_args = parser.parse_args()



GLUE_TASKS = ["cola","mnli","mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
model_checkpoint = "mistralai/Mistral-7B-v0.1"  
batch_size = 4
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
for t in GLUE_TASKS :
    print('######## '+t+' #######' )
    task = t
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map = 'auto'
        )
    model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, peft_config)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}--AdamWr-lr2e-4",
        save_strategy = "epoch",
        learning_rate=4e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        metric_for_best_model=metric_name
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)


    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = Trainer( 
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        custom_optimizer = cmd_args.optimizer
    )

    trainer.train()

    print('task name: ', t, ' eval result: ', trainer.evaluate())
