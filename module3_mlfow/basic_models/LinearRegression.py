
# to avoid some obscure warnings https://github.com/mlflow/mlflow/issues/8605
# import pkg_resources  # part of setuptools
# version = pkg_resources.get_distribution("pip").version

# https://www.restack.io/docs/mlflow-knowledge-mlflow-userwarning-setuptools-distutils
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

import mlflow

#mlflow.set_tracking_uri("http://0.0.0.0:5001/")
mlflow.set_tracking_uri("http://172.17.0.2:5000/")

# import os
# import sys
# from pathlib import Path
# ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
# print(ROOT_DIR)
# sys.path.append(os.path.dirname(ROOT_DIR))
from utils import default_mlflow_run

RANDOM_SEED = 42

# import data
X = pd.read_csv('housing.csv')
y = X.pop('median_house_value')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
                                                    random_state = 42)


# Create experiment

# try:
#     experiment_description = (
#         'Trying basic models with default parameters and no feature engineering.'
#     )

#     experiment_tags = {
#         'env': 'dev',
#         'project_name': 'California_housing',
#         'mlflow.note.content': experiment_description,
#         'pipeline_version': '1.0'
#     }

#     exp_id = mlflow.create_experiment(
#         name='Basics_proj',
#         tags=experiment_tags
#     )
# except:
#     pass

exp_id = mlflow.get_experiment_by_name('Basic_Models').experiment_id

# Run experiments
default_mlflow_run(LinearRegression(),
                   'lr_func',# exp_id,
                   X_train_raw, X_test_raw, y_train, y_test)

# default_mlflow_run(HistGradientBoostingRegressor(random_state=RANDOM_SEED),
#                    'hgb_func',# exp_id,
#                    X_train_raw, X_test_raw, y_train, y_test)

# default_mlflow_run(RandomForestRegressor(random_state=RANDOM_SEED),
#                    'rfr_func',# exp_id,
#                    X_train_raw, X_test_raw, y_train, y_test)