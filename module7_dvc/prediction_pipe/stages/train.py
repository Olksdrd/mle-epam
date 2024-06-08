import numpy as np
import pandas as pd
import pickle
import yaml
import os
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingRegressor


def train_model(cls):
    """
    Train HGB model and saves the training pipeline
    as a pickle file and as an mlflow model.
    """
    X_train = pd.read_csv('../data/split/data_train.csv')
    y_train = X_train.pop('median_house_value')

    cls.fit(X_train, y_train)

    path = './models'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f'{path}/cls.pkl', 'wb') as f:
        pickle.dump(cls, f)


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['training']

    num_union = FeatureUnion([
        ('identity', 'passthrough'),
        ('binning', make_pipeline(KBinsDiscretizer(strategy='kmeans',
                                                   n_bins=params['n_bins']),
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
        HistGradientBoostingRegressor(
            random_state=params['random_seed'],
            max_depth=params['max_depth'],
            l2_regularization=params['l2_regularization'])
    )

    train_model(cls)
