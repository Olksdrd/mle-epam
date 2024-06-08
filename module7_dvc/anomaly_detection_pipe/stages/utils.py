import pandas as pd
import os


def detect_outliers(cls):
    """Train and save an outlier detector for future use"""

    X_train = pd.read_csv('../data/cleaned/X_train_clean.csv')
    X_val = pd.read_csv('../data/cleaned/X_val_clean.csv')

    cls.fit(X_train)

    outliers = pd.Series(cls.predict(X_val))

    path = './outliers'
    if not os.path.exists(path):
        os.mkdir(path)

    outliers.to_csv(f'{path}/{type(cls).__name__}_outliers.csv', index=False)
