import numpy as np
import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(pipeline):
    """Preprocess data for anomaly detection"""
    X_train = pd.read_csv('../data/split/data_train.csv')
    X_train.pop('median_house_value')
    X_val = pd.read_csv('../data/split/data_val.csv')
    X_val.pop('median_house_value')

    X_train_clean = pipeline.fit_transform(X_train)
    X_val_clean = pipeline.transform(X_val)

    path = '../data/cleaned'
    if not os.path.exists(path):
        os.mkdir(path)

    pd.DataFrame(X_train_clean).to_csv(f'{path}/X_train_clean.csv', index=False)
    pd.DataFrame(X_val_clean).to_csv(f'{path}/X_val_clean.csv', index=False)


if __name__ == '__main__':
    preprocessor = ColumnTransformer([
        ('cat',
         make_pipeline(
            SimpleImputer(strategy='most_frequent'), OneHotEncoder()),
            make_column_selector(dtype_include='object')
         ),

        ('num',
            SimpleImputer(strategy='median'),
            make_column_selector(dtype_include=np.number)
         )
    ])

    preprocess_data(preprocessor)
