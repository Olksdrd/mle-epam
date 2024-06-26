import json
import requests
import os
import sys

sys.path.insert(0, os.getcwd())
import utils.configs as configs
import utils.funcs as f


_, lets_pretend_this_is_new_data, _, _ = f.load_data(configs.DATA_PATH,
                                                     random_state=configs.RANDOM_STATE)

try:
    batch_size = int(sys.argv[1])
except:
    batch_size = 40

batch = lets_pretend_this_is_new_data.tail(batch_size)
data = {"dataframe_split": batch.to_dict(orient="split")}

headers = {'Content-Type': 'application/json'}
endpoint = configs.MLFLOW_SERVING_URI

r = requests.post(endpoint, data=json.dumps(data), headers=headers)
print(r.status_code)
print(r.text)