from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import numpy as np
import evaluate
import json
import pandas as pd

with open('data.json','r') as file:
    data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(data["model"],use_safetensors=data["use_tensor"])
model = AutoModelForSequenceClassification.from_pretrained(data["model"],use_safetensors=data["use_tensor"])

df = pd.read_csv(data["csv"])

dataset = Dataset.from_pandas(df)

def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True,max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="test_trainer")


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

new_tokenized_datasets =  tokenized_datasets.train_test_split(test_size=0.2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=new_tokenized_datasets['train'],
    eval_dataset=new_tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(data["model_to_save"])
model.save_pretrained(data["model_to_save"])
tokenizer.save_pretrained(data["model_to_save"])