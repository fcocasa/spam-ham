import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json

with open('data.json','r') as file:
    data = json.load(file)

with open(data['example_file']) as file:
    texto = file.read()

# tokenizer = AutoTokenizer.from_pretrained("skandavivek2/spam-classifier")
# model = AutoModelForSequenceClassification.from_pretrained("skandavivek2/spam-classifier")
tokenizer = AutoTokenizer.from_pretrained(data["model"],use_safetensors=data["use_tensor"])
model = AutoModelForSequenceClassification.from_pretrained(data["model"],use_safetensors=data["use_tensor"])


inputs = tokenizer(texto, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

prob = F.softmax(logits, dim=1)


if predicted_class_id==1:
    print('Es Spam con probabilidad:', prob[0][1].item()*100)
else:
    print('No es Spam con probabilidad:',prob[0][0].item()*100)
    
