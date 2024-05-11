import os
import unittest
import pickle

from sklearn.utils.validation import check_is_fitted

module5_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(f'{module5_path}/models/cls.pkl', 'rb') as f:
            cls.classifier = pickle.load(f)
        with open(f'{module5_path}/models/outlier_detector.pkl', 'rb') as f:
            cls.outlier_detector = pickle.load(f)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_cls_fitted(self):
        self.assertEqual(check_is_fitted(TestModel.classifier), None)

    def test_outlier_detector_fitted(self):
        self.assertEqual(check_is_fitted(TestModel.outlier_detector), None)


if __name__ == '__main__':
    unittest.main()
