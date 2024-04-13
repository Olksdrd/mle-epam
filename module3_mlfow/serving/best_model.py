import warnings
warnings.filterwarnings("ignore")

import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

sys.path.insert(0, os.getcwd())
import utils.configs as configs
import utils.funcs as f

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils import estimator_html_repr



def final_run(regressor, run_name,
         X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:

        pipe = Pipeline([
            ('interactions', f.FeatureInteractions()),
            ('preprocessing', f.preprocessor_v2),
            ('regressor', regressor)
        ])

        model_name = type(pipe['regressor']).__name__

        print(f'Logging {model_name} model...')

        pipe.fit(X_train, y_train)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = f.eval_model(pipe, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        estimator = estimator_html_repr(pipe)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, 'my_estimator.html')
            path.write_text(estimator, encoding="utf-16")

            mlflow.log_artifact(path)

        model_signature = infer_signature(X_train, y_train)
        
        mlflow.sklearn.log_model(pipe,
                                artifact_path=model_name,
                                signature=model_signature,
                                registered_model_name=model_name,
                                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)




if __name__ == '__main__':
    # Configure mlflow
    # mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI')) # just in case if needed
    client = MlflowClient()

    # Load data
    X_train_raw, X_test_raw, y_train, y_test = f.load_data(configs.DATA_PATH,
                                                           random_state=configs.RANDOM_STATE)

    # Create experiment
    try:
        exp_id = mlflow.create_experiment(name='Best_Model')
    except:
        exp_id = mlflow.get_experiment_by_name('Best_Model').experiment_id


    all_runs = mlflow.search_runs(search_all_experiments=True, filter_string="metrics.test_r2 > 0.8")

    l2 = all_runs.sort_values('metrics.test_r2', ascending=False)['params.l2_regularization'].head(1)
    max_depth = all_runs.sort_values('metrics.test_r2', ascending=False)['params.max_depth'].head(1)


    hbg = HistGradientBoostingRegressor(random_state=configs.RANDOM_STATE,
                                        max_depth=int(max_depth),
                                        l2_regularization=float(l2))

    final_run(hbg, 'hgb_best', X_train_raw, X_test_raw, y_train, y_test)

    
    model_name = 'HistGradientBoostingRegressor'
    model_ver = client.get_latest_versions(model_name)[0].version

    client.update_model_version(name=model_name,
                                version=model_ver,
                                description="Model with the best R2")

    client.set_registered_model_alias(model_name,
                                      alias='prod',
                                      version=model_ver)

    client.set_model_version_tag(name=model_name,
                                 version=model_ver,
                                 key='tuned', value='yes')
    
    client.set_model_version_tag(name=model_name,
                                 version=model_ver,
                                 key='features', value='v2')