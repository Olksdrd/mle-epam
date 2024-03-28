import os
import sys
from pathlib import Path
import json

import pandas as pd
import opendatasets as od

# Set path to root directory of the project
ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(os.path.dirname(ROOT_DIR))

# Load config file
CONF_FILE = os.path.abspath('settings.json') 
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Paths to data directories
RAW_DATA_DIR = os.path.abspath(conf['general']['raw_data_dir'])
PROCESSED_DATA_DIR = os.path.abspath(conf['general']['processed_data_dir'])


def create_folders():
    """Creates folder system to store raw and processed data"""
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)


def load_dataset(URL):
    """Loads dataset from Kaggle URL and cleans up all temporary files"""
    dataset_name = URL.split('/')[-1]

    od.download(URL)

    Path(dataset_name).rename('temp')
    Path('temp/IMDB Dataset.csv').rename('temp/IMDB_dataset.csv')
    os.rename('temp/IMDB_dataset.csv', 'data/raw_data/IMDB_dataset.csv')
    os.rmdir('temp')


URL = conf['general']['kaggle_url']


if __name__ == '__main__':
    create_folders()
    load_dataset(URL)

    path_to_raw_data = 'data/raw_data/IMDB_dataset.csv'
    df = pd.read_csv(path_to_raw_data)
    print(df.head())