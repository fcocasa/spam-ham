import evaluate
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

with open('data.json','r') as file:
    data = json.load(file)

df = pd.read_csv(data['csv_test'])

# tokenizer = AutoTokenizer.from_pretrained("skandavivek2/spam-classifier")
# model = AutoModelForSequenceClassification.from_pretrained("skandavivek2/spam-classifier")

tokenizer = AutoTokenizer.from_pretrained(data["model"],use_safetensors=data["use_tensor"])
model = AutoModelForSequenceClassification.from_pretrained(data["model"],use_safetensors=data["use_tensor"])


labels = []
predictions = []

for line in df[['text','label']].dropna().to_dict(orient='records'):
    inputs = tokenizer(line['text'], return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    predictions += [predicted_class_id]
    labels += [line['label']]

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
result = clf_metrics.compute(predictions=predictions, references=labels)
print(result)