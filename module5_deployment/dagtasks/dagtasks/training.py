import os

import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error)

import mlflow


def _eval_model(model, X_train, X_test, y_train, y_test):
    """Calculate R2, RMSE and MAE for train and test sets."""
    train_predicted = model.predict(X_train)
    test_predicted = model.predict(X_test)

    return {'train_r2': r2_score(y_train, train_predicted),
            'train_rmse': mean_squared_error(y_train,
                                             train_predicted,
                                             squared=False),
            'train_mae': mean_absolute_error(y_train, train_predicted),
            'test_r2': r2_score(y_test, test_predicted),
            'test_rmse': mean_squared_error(y_test,
                                            test_predicted,
                                            squared=False),
            'test_mae': mean_absolute_error(y_test, test_predicted)}


num_union = FeatureUnion([
    ('identity', 'passthrough'),
    ('binning', make_pipeline(KBinsDiscretizer(strategy='kmeans'),
                              VarianceThreshold(threshold=0.1)))
])

num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    num_union
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder()
)

preprocessor = ColumnTransformer([
    ('cat', cat_pipe, make_column_selector(dtype_include='object')),
    ('num', num_pipe, make_column_selector(dtype_include=np.number))
])

cls = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(random_state=42)
)

outlier_detector = make_pipeline(
    preprocessor,
    IsolationForest(contamination=0.05, random_state=42)
)


def _load_datasets():
    X_train = pd.read_csv('data/data_train.csv')
    X_val = pd.read_csv('data/data_val.csv')
    y_train = X_train.pop('median_house_value')
    y_val = X_val.pop('median_house_value')

    return X_train, X_val, y_train, y_val


def detect_outliers():
    """Train and save an outlier detector for future use"""
    X_train, X_val, _, _ = _load_datasets()

    outlier_detector.fit(X_train)

    outlier_prorportion = (outlier_detector.predict(X_val) == -1).mean()
    print('% of outliers in training set = 5%')
    print(f'% of outliers in validation set = {outlier_prorportion*100:.2f}%')

    path = './models'
    if not os.path.exists(path):
        os.mkdir(path)

    with open('models/outlier_detector.pkl', 'wb') as f:
        pickle.dump(outlier_detector, f)


def train_model():
    """
    Train HGB model and saves the training pipeline
    as a pickle file and as an mlflow model.
    """
    X_train, X_val, y_train, y_val = _load_datasets()

    with mlflow.start_run(run_name='hgb', experiment_id=0):

        model_name = 'HistGradientBoosting'

        print(f'Logging {model_name} model...')

        cls.fit(X_train, y_train)

        mlflow.log_params(cls['histgradientboostingregressor'].get_params())

        metrics = _eval_model(cls, X_train, X_val, y_train, y_val)
        print(metrics)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.save_model(
            cls,
            path='./models/hgb',
            # artifact_path=model_name,
            serialization_format='cloudpickle'
            )

        with open('models/cls.pkl', 'wb') as f:
            pickle.dump(cls, f)


def validate_model():
    with open('models/cls.pkl', 'rb') as f:
        cls = pickle.load(f)

    X = pd.read_csv('data/housing_clean.csv')
    y = X.pop('median_house_value')

    spliter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    res = cross_val_score(cls, X, y, cv=spliter)

    print('CV results (R2): ', list(res))
    print('Mean R2 score: ', res.mean())
