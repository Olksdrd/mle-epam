import os
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


def str_to_num():
    """Converts str datatype to numeric"""
    df = pd.read_csv('data/housing_districts.csv')
    df['total_bedrooms'] = df['total_bedrooms'].astype(float)
    df.to_csv('data/housing_districts.csv', index=False)


def uniform_missing_indicator():
    """Encodes all numeric missing values as np.nan"""
    df = pd.read_csv('data/housing_districts.csv')
    df.replace(-999, np.nan, inplace=True)
    df.to_csv('data/housing_districts.csv', index=False)


def rename_unknown_category():
    """Encodes all categorical NAs as 'Missing' category"""
    df = pd.read_csv('data/housing_geo.csv')
    df.replace('-999', 'Missing', inplace=True)
    df.to_csv('data/housing_geo.csv', index=False)


def merge_tables():
    """Joins a dataset into one table and deletes two original tables"""
    df_geography = pd.read_csv('data/housing_geo.csv')
    df_districts = pd.read_csv('data/housing_districts.csv')

    merged_df = df_geography.merge(df_districts,
                                   left_index=True,
                                   right_index=True)
    merged_df.to_csv('data/housing_clean.csv', index=False)

    print('Joined table saved to data/housing_clean.csv')

    os.remove('data/housing_geo.csv')
    os.remove('data/housing_districts.csv')


# class ColumnArranger(BaseEstimator, TransformerMixin):
#     """
#     Make columns consistent between training and production data:
#     - remove columns unseen during training.
#     - add all missing columns that were present during training.
#     - make sure that column order is always the same.
#     This ensures that the pipeline outputs the exact same
#     set of labels at each stage of the model life cycle.
#     """
#     def fit(self, X, y=None):
#         self.columns = X.columns
#         return self

#     def transform(self, X):
#         X.copy()  # so we don't modify original input
#         new_cols = X.columns

#         # # remove columns unseen during training
#         # unseen_cols = np.setdiff1d(new_cols, self.columns)
#         # X = X.drop(unseen_cols, axis=1)

#         # add missing columns
#         missing_cols = np.setdiff1d(self.columns, new_cols)
#         X[missing_cols] = np.nan

#         # make sure column order is the same as during fitting
#         return X[self.columns]


# class DtypeConverter(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         self.dtypes = X.dtypes
#         return self

#     def transform(self, X):
#         dtypes_map = {col: dtype for col, dtype in zip(X.columns, self.dtypes)}
#         try:
#             return X.astype(dtypes_map)
#         except ValueError:
#             raise Exception("Can't convert dtypes")


# def save_columns_signature():

#     X = pd.read_csv('data/housing_clean.csv')
#     X.pop('median_house_value')

#     col_signature = make_pipeline(
#         ColumnArranger(),
#         DtypeConverter()
#     )
#     col_signature.fit(X)

#     path = './models'
#     if not os.path.exists(path):
#         os.mkdir(path)

#     with open('models/col_signature.pkl', 'wb') as f:
#         pickle.dump(col_signature, f)
