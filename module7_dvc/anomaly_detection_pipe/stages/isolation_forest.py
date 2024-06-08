import yaml
from utils import detect_outliers
from sklearn.ensemble import IsolationForest


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['isolation_forest']
    cls = IsolationForest(
        contamination=params['contamination'],
        random_state=params['random_seed']
        )
    detect_outliers(cls)
