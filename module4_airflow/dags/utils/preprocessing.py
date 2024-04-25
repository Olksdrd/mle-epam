import pandas as pd

from sklearn.model_selection import train_test_split


def feature_interactions():
    df = pd.read_csv('data/housing_clean.csv')

    df['lat*long'] = df['latitude'] * df['longitude']
    df['lat/long'] = df['latitude'] / df['longitude']
    df['rooms_per_house'] = df['total_rooms'] / df['households']
    df['bedrooms_per_house'] = df['total_bedrooms'] / df['households']
    df['rooms_per_pop'] = df['total_rooms'] / df['population']
    df['bedrooms_per_pop'] = df['total_bedrooms'] / df['population']
    df['bedroom_frac'] = df['total_bedrooms'] / df['total_rooms']
    df['pop_per_house'] = df['population'] / df['households']

    df['rooms_per_income'] = df['total_rooms'] / df['median_income']
    df['bedrooms_per_income'] = df['total_bedrooms'] / df['median_income']

    df.to_csv('data/housing_clean.csv', index=False)


def train_val_split():
    df = pd.read_csv('data/housing_clean.csv')

    df_train, df_val = train_test_split(df, random_state=42)

    df_train.to_csv('data/data_train.csv', index=False)
    df_val.to_csv('data/data_val.csv', index=False)
