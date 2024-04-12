# to avoid some obscure warnings 
# https://www.restack.io/docs/mlflow-knowledge-mlflow-userwarning-setuptools-distutils
import warnings
warnings.filterwarnings("ignore")

import os
import sys

from sklearn.ensemble import HistGradientBoostingRegressor

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, os.getcwd())
import utils.configs as configs
import utils.funcs as f



if __name__ == '__main__':
    # Configure mlflow
    # mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI')) # just in case if needed
    client = MlflowClient()

    # Load data
    X_train_raw, X_test_raw, y_train, y_test = f.load_data(configs.DATA_PATH,
                                                           random_state=configs.RANDOM_STATE)

    # Run experiments
    f.pipe_ver2_mlflow_run(HistGradientBoostingRegressor(random_state=configs.RANDOM_STATE),
                         X_train_raw, X_test_raw, y_train, y_test)

    # Set tags/aliases/descriptions
    model_name = 'HistGradientBoostingRegressor'
    model_ver = client.get_latest_versions(model_name)[0].version

    client.update_model_version(name=model_name,
                                version=model_ver,
                                description='Some new features added; default params')

    client.set_model_version_tag(name=model_name,
                                version=model_ver,
                                key='features', value='v2')
    
    client.set_model_version_tag(name=model_name,
                                version=model_ver,
                                key='tuned', value='no')

    client.set_registered_model_alias(model_name,
                                      alias='defaultv2',
                                      version=model_ver)