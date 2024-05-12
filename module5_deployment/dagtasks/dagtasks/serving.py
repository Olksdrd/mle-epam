import os
from os.path import isfile, join
from datetime import datetime
import pickle

import pandas as pd


def get_batch(batch_size):
    """Simulate getting a batch of new data"""
    # Create necessary folders
    paths = ['./results', './results/batch_new', './results/batch_pred']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    # Generate new batch
    batch = pd.read_csv('data/data_val.csv').sample(batch_size)
    batch.pop('median_house_value')

    # Save the batch
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = f'batch_{current_date_time}'
    batch.to_csv(f'results/batch_new/{file_name}.csv', index=False)


def check_for_outliers():
    """Calculate outlier percentage for each new batch"""
    # Load the model
    with open('models/outlier_detector.pkl', 'rb') as f:
        model = pickle.load(f)

    # Generate predictions for each new batch
    path = './results/batch_new'
    batches_to_predict = [f for f in os.listdir(path) if isfile(join(path, f))]
    if batches_to_predict:
        for batch_name in batches_to_predict:
            batch = pd.read_csv(f'{path}/{batch_name}')

            outlier_res = model.predict(batch)
            outlier_pct = (outlier_res == -1).mean()

            if outlier_pct > 0.2:
                print('Too many outliers')
            else:
                print(f'{outlier_pct*100:.2f}% of outliers in {batch_name}')


def get_predictions():
    """Generate predictions for each new batch of data"""
    # Load the model
    with open('models/hgb/model.pkl', 'rb') as f:
        cls = pickle.load(f)

    # Generate predictions for each new batch
    path = './results/batch_new'
    batches_to_predict = [f for f in os.listdir(path) if isfile(join(path, f))]
    if batches_to_predict:
        for batch_name in batches_to_predict:
            batch = pd.read_csv(f'{path}/{batch_name}')

            batch['median_house_value_pred'] = cls.predict(batch)
            batch.to_csv(f'results/batch_pred/{batch_name}', index=False)

            os.remove(f'results/batch_new/{batch_name}')
