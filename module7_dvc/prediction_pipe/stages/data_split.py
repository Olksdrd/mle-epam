import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os


def train_val_split(random_state, test_size):
    df = pd.read_csv('../data/housing_clean.csv')

    df_train, df_val = train_test_split(df,
                                        random_state=random_state,
                                        test_size=test_size)

    path = '../data/split'
    if not os.path.exists(path):
        os.mkdir(path)

    df_train.to_csv(f'{path}/data_train.csv', index=False)
    df_val.to_csv(f'{path}/data_val.csv', index=False)


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['data_split']

    train_val_split(random_state=params['random_seed'],
                    test_size=params['test_size'])
