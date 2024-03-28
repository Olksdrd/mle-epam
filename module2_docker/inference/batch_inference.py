import os
import sys
import json
from pathlib import Path

import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.abspath('settings.json') 
with open(CONF_FILE, "r") as file:
    conf = json.load(file)


df = pd.read_csv(conf['general']['dataset_path'])
texts = list(df['review'])
labels = [0 if i=='negative' else 1 for i in list(df['sentiment'])]


transformer_name = conf['general']['model_name']

model = AutoModelForSequenceClassification.from_pretrained(transformer_name)
tokenizer = AutoTokenizer.from_pretrained(transformer_name)

batch_size = conf['inference']['batch_size']
X_train = texts[:batch_size]
batch = tokenizer(X_train,
                  padding=True,
                  truncation=True,
                  max_length=512,
                  return_tensors='pt'
                  )

with torch.no_grad():
  outputs = model(**batch, labels=torch.tensor(labels[:batch_size]))
  print(outputs)
  predictions = F.softmax(outputs.logits, dim=1)
  print(predictions)
  labels = torch.argmax(predictions, dim=1)
  print(labels)
  labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
  print(labels)