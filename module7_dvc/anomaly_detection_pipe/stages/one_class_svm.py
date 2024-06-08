import yaml
from utils import detect_outliers
from sklearn.svm import OneClassSVM


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['one_class_svm']
    cls = OneClassSVM(gamma=params['gamma'])
    detect_outliers(cls)
