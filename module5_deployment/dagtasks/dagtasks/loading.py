import os
import requests

import numpy as np
import pandas as pd


def load_data():
    """Loads California housing dataset from github"""
    path = './data'
    if not os.path.exists(path):
        os.mkdir(path)

    dataset_name = 'housing.csv'
    URL = 'https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv'
    r = requests.get(URL)
    open(f'data/{dataset_name}', 'wb').write(r.content)


def read_data():
    """
    Makes the dataset not so clean, so the pipeline is more interesting:
    - converts total_bedrooms to str.
    - adds NAs as -999.
    - splits into two tablws.
    """
    df = pd.read_csv('data/housing.csv')

    df['total_bedrooms'] = df['total_bedrooms'].astype(str)

    filt = np.random.uniform(0, 1, size=df.shape[0]) > 0.99
    df.loc[filt, 'total_rooms'] = -999

    filt2 = np.random.uniform(0, 1, size=df.shape[0]) > 0.85
    df.loc[filt2, 'ocean_proximity'] = '-999'

    df_geography = df[['longitude', 'latitude', 'ocean_proximity']]
    df_rest = df.drop(['longitude', 'latitude', 'ocean_proximity'], axis=1)

    df_geography.to_csv('data/housing_geo.csv', index=False)
    df_rest.to_csv('data/housing_districts.csv', index=False)

    print('''Data saved to:
          - data/housing_geo.csv
          - data/housing_districts.csv'''
          )

    os.remove('data/housing.csv')
