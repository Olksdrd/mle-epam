import pandas as pd


def feature_interactions():
    """Generates new features by combining existing ones"""
    df = pd.read_csv('../data/housing.csv')

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

    df.to_csv('../data/housing_clean.csv', index=False)

    print('Feature interactions added.')


if __name__ == '__main__':
    feature_interactions()
