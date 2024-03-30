import os
import sys
import json
from pathlib import Path
import logging
import logging.config
from time import time

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


def configure_logs():
    LOG_FILE = os.path.abspath(conf['general']['logs_config']) 
    with open(LOG_FILE, 'r') as file:
        log_dict = json.load(file)

    logging.config.dictConfig(log_dict)


df = pd.read_csv(conf['general']['dataset_path'])
texts = list(df['review'])
labels = [0 if i=='negative' else 1 for i in list(df['sentiment'])]


transformer_name = conf['general']['model_name']

model = AutoModelForSequenceClassification.from_pretrained(transformer_name)
tokenizer = AutoTokenizer.from_pretrained(transformer_name)


start_time = time()
configure_logs()

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
  predictions = F.softmax(outputs.logits, dim=1)
  labels = torch.argmax(predictions, dim=1)
  # print(labels)
  text_labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
  print(text_labels)

finish_time = time()
time_delta = finish_time - start_time
logging.info(f'Inference for {batch_size} datapoints took {time_delta:.2f} seconds.')
logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)