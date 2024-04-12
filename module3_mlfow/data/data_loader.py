import os
import sys
from pathlib import Path

import opendatasets as od

sys.path.insert(0, os.getcwd())
import utils.configs as configs


def load_dataset(URL):
    """Loads dataset from Kaggle URL and cleans up all temporary files"""
    ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
    sys.path.append(os.path.dirname(ROOT_DIR))
    # sys.path.append(configs.WORK_DIR) # alternative approach


    dataset_name = URL.split('/')[-1]
    od.download(URL)
    Path(dataset_name).rename('temp')
    os.rename('temp/housing.csv', configs.DATA_PATH)
    os.rmdir('temp')



if __name__ == '__main__':
    load_dataset(configs.DATASET_URL)
    print(f'Dataset loaded to {configs.DATA_PATH}')