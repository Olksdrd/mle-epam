import os

import numpy as np
import pandas as pd


def str_to_num():
    df = pd.read_csv('data/housing_districts.csv')
    df['total_bedrooms'] = df['total_bedrooms'].astype(float)
    df.to_csv('data/housing_districts.csv', index=False)


def uniform_missing_indicator():
    df = pd.read_csv('data/housing_districts.csv')
    df.replace(-999, np.nan, inplace=True)
    df.to_csv('data/housing_districts.csv', index=False)


def rename_unknown_category():
    df = pd.read_csv('data/housing_geo.csv')
    df.replace('-999', 'Missing', inplace=True)
    df.to_csv('data/housing_geo.csv', index=False)


def merge_tables():
    df_geography = pd.read_csv('data/housing_geo.csv')
    df_districts = pd.read_csv('data/housing_districts.csv')
    print(df_geography.shape)
    print(df_geography.head())
    print(df_districts.shape)
    print(df_districts.head())
    merged_df = df_geography.merge(df_districts,
                                   left_index=True,
                                   right_index=True)
    merged_df.to_csv('data/housing_clean.csv', index=False)

    os.remove('data/housing_geo.csv')
    os.remove('data/housing_districts.csv')
