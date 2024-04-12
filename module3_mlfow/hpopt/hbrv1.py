import warnings
warnings.filterwarnings("ignore")
import os
import sys

from functools import partial
from hyperopt import fmin, tpe, Trials, hp

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

import mlflow

sys.path.insert(0, os.getcwd())
import utils.configs as configs
import utils.funcs as f


# Configure mlflow
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI')) # just in case if needed


# Load data
X_train_raw, X_test_raw, y_train, y_test = f.load_data(configs.DATA_PATH,
                                                       random_state=configs.RANDOM_STATE)


def objective_function(params, X_train, X_test, y_train, y_test):
    
    params.update({'regressor__max_depth': int(params['regressor__max_depth'])})
    params.update({'regressor__l2_regularization': float(params['regressor__l2_regularization'])})

    pipe = Pipeline([
        ('preprocessing', f.preprocessor_v1),
        ('regressor', HistGradientBoostingRegressor(random_state=configs.RANDOM_STATE))
    ])


    pipe.set_params(**params)

    with mlflow.start_run(nested=True) as run:
        pipe.fit(X_train, y_train)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = f.eval_model(pipe, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, f'{run.info.run_id}-hbr')

    return metrics['test_rmse']



search_space = {
    'regressor__max_depth': hp.quniform('regressor__max_depth', 20, 200, 10),
    'regressor__l2_regularization': hp.quniform('regressor__l2_regularization', 0, 10, 1)
}


with mlflow.start_run() as run:

    best_params = fmin(
        fn=partial(objective_function, 
                   X_train=X_train_raw,
                   X_test=X_test_raw,
                   y_train=y_train,
                   y_test=y_test
        ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=configs.HPOPT_NUM_TRIALS,
        trials=Trials()
    )

    pipe = Pipeline([
        ('preprocessing', f.preprocessor_v1),
        ('regressor', HistGradientBoostingRegressor(random_state=configs.RANDOM_STATE))
    ])

    best_params.update({'regressor__max_depth': int(best_params['regressor__max_depth'])})
    best_params.update({'regressor__l2_regularization': float(best_params['regressor__l2_regularization'])})

    pipe.set_params(**best_params)

    pipe.fit(X_train_raw, y_train)
    preds = pipe.predict(X_test_raw)

    mlflow.log_params(pipe['regressor'].get_params())

    metrics = f.eval_model(pipe, X_train_raw, X_test_raw, y_train, y_test)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, f'{run.info.run_id}-hbr')
