import pickle
import pandas as pd
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             root_mean_squared_error)

from dvclive import Live


def validate_model(metrics_path):
    with open('models/cls.pkl', 'rb') as f:
        cls = pickle.load(f)

    with Live(dir=metrics_path) as live:
        X_train = pd.read_csv('../data/split/data_train.csv')
        y_train = X_train.pop('median_house_value')
        X_val = pd.read_csv('../data/split/data_val.csv')
        y_val = X_val.pop('median_house_value')

        train_predicted = cls.predict(X_train)
        val_predicted = cls.predict(X_val)

        live.log_metric("train/MAE",
                        mean_absolute_error(y_train, train_predicted),
                        plot=False)

        live.log_metric("train/RMSE",
                        root_mean_squared_error(y_train, train_predicted),
                        plot=False)

        live.log_metric("train/R2",
                        r2_score(y_train, train_predicted),
                        plot=False)

        live.log_metric("val/MAE",
                        mean_absolute_error(y_val, val_predicted),
                        plot=False)

        live.log_metric("val/RMSE",
                        root_mean_squared_error(y_val, val_predicted),
                        plot=False)

        live.log_metric("val/R2",
                        r2_score(y_val, val_predicted),
                        plot=False)


if __name__ == '__main__':
    EVAL_PATH = "eval"
    validate_model(EVAL_PATH)
