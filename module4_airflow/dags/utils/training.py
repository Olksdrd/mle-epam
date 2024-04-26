import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error)


def eval_model(true, predicted):
    r2 = r2_score(true, predicted)
    rmse = mean_squared_error(true, predicted, squared=False)
    mae = mean_absolute_error(true, predicted)

    print(f'R2  : {r2:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE : {mae:.2f}')


num_union = FeatureUnion([
    ('identity', 'passthrough'),
    ('binning', make_pipeline(KBinsDiscretizer(strategy='kmeans'),
                              VarianceThreshold(threshold=0.1)))
])

num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    num_union
)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object')),
    ('num', num_pipe, make_column_selector(dtype_include=np.number))
])

cls = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(random_state=42)
)


def train_model():
    """Trains HGB model and saves the training pipeline"""
    X_train = pd.read_csv('data/data_train.csv')
    X_val = pd.read_csv('data/data_val.csv')
    y_train = X_train.pop('median_house_value')
    y_val = X_val.pop('median_house_value')

    cls.fit(X_train, y_train)
    eval_model(y_val, cls.predict(X_val))

    with open('data/cls.pkl', 'wb') as f:
        pickle.dump(cls, f)


def validate_model():
    with open('data/cls.pkl', 'rb') as f:
        cls = pickle.load(f)

    X = pd.read_csv('data/housing_clean.csv')
    y = X.pop('median_house_value')

    spliter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    res = cross_val_score(cls, X, y, cv=spliter)

    print('CV results (R2): ', list(res))
    print('Mean R2 score: ', res.mean())
