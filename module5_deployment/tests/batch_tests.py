import os
from os.path import isfile, join
import unittest

import pandas as pd
import numpy as np

module5_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(f'{module5_path}/data/data_train.csv')
        cls.df.pop('median_house_value')

        path = f'{module5_path}/results/batch_new'
        batch_name = [f for f in os.listdir(path) if isfile(join(path, f))][0]
        batch_path = f'{module5_path}/results/batch_new/{batch_name}'
        cls.batch = pd.read_csv(batch_path)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_columns(self):
        test = np.all(TestBatch.batch.columns == TestBatch.df.columns)
        self.assertEqual(test, True)

    def test_dtypes(self):
        test = np.all(TestBatch.batch.dtypes == TestBatch.df.dtypes)
        self.assertEqual(test, True)


if __name__ == '__main__':
    unittest.main()
