import os

import numpy as np
import pandas as pd


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
