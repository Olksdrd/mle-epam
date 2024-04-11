import warnings
warnings.filterwarnings("ignore")

from functools import partial
from typing import Dict, List
from hyperopt import fmin, tpe, Trials, hp
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

import mlflow

from utils import preprocessor_trees, eval_model, FeatureInteractions, BinFeatures

#mlflow.set_tracking_uri("http://0.0.0.0:5001/")
mlflow.set_tracking_uri("http://172.17.0.2:5000/")

RANDOM_SEED = 42

# import data
X = pd.read_csv('housing.csv')
y = X.pop('median_house_value')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
                                                    random_state=RANDOM_SEED)


def objective_function(params: Dict, 
                       X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.DataFrame, y_test: pd.DataFrame):
    
    params.update({'regressor__max_depth': int(params['regressor__max_depth'])})
    params.update({'regressor__l2_regularization': float(params['regressor__l2_regularization'])})

    pipe = Pipeline([
        ('interactions', FeatureInteractions()),
        ('binning', BinFeatures(['population', 'median_income'])),
        ('preprocessing', preprocessor_trees),
        ('regressor', HistGradientBoostingRegressor(random_state=RANDOM_SEED))
    ])


    pipe.set_params(**params)

    with mlflow.start_run(
        #experiment_id=exp_id, 
        nested=True) as run:
        pipe.fit(X_train, y_train)
        # preds = pipe.predict(X_test)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = eval_model(pipe, X_train_raw, X_test_raw, y_train, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, f'{run.info.run_id}-hbr')

    return metrics['test_rmse'] # we're going to minimize it, so need minus for r2, accuracy, f1-score etc



search_space = {
    'regressor__max_depth': hp.quniform('regressor__max_depth', 10, 150, 5),
    'regressor__l2_regularization': hp.quniform('regressor__l2_regularization', 0, 5, 1)
}


with mlflow.start_run(
    # run_name='HpTuning', experiment_id=exp_id
    ) as run:

    best_params = fmin(
        fn=partial(objective_function, 
                   X_train=X_train_raw,
                   X_test=X_test_raw,
                   y_train=y_train,
                   y_test=y_test
        ),
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials()
    )

    pipe = Pipeline([
        ('interactions', FeatureInteractions()),
        ('binning', BinFeatures(['population', 'median_income'])),
        ('preprocessing', preprocessor_trees),
        ('regressor', HistGradientBoostingRegressor(random_state=RANDOM_SEED))
    ])

    best_params.update({'regressor__max_depth': int(best_params['regressor__max_depth'])})
    best_params.update({'regressor__l2_regularization': float(best_params['regressor__l2_regularization'])})

    pipe.set_params(**best_params)

    pipe.fit(X_train_raw, y_train)
    preds = pipe.predict(X_test_raw)

    mlflow.log_params(pipe['regressor'].get_params())

    metrics = eval_model(pipe, X_train_raw, X_test_raw, y_train, y_test)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, f'{run.info.run_id}-hbr')
