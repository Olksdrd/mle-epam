import json
import requests

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('housing.csv')

X = df.copy()
y = X.pop('median_house_value')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
                                                    random_state = 42)


batch = X_test_raw.tail(40)
data = {"dataframe_split": batch.to_dict(orient="split")}

headers = {'Content-Type': 'application/json'}
# endpoint = 'http://127.0.0.1:5002/invocations'
endpoint = 'http://0.0.0.0:5002/invocations'

r = requests.post(endpoint, data=json.dumps(data), headers=headers)
print(r.status_code)
print(r.text)