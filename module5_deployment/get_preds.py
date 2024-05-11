import json
import requests
from time import sleep
import pandas as pd

headers = {'Content-Type': 'application/json'}
endpoint = 'http://0.0.0.0:5002/invocations'

for i in range(10):
    df = pd.read_csv('data/data_val.csv').sample(1)
    data = {"dataframe_split": df.to_dict(orient="split")}
    r = requests.post(endpoint, data=json.dumps(data), headers=headers)
    print(r.status_code)
    print(r.text)
    print(r.json()['predictions'][0])
    sleep(2)
