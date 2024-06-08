import yaml
from utils import detect_outliers
from sklearn.covariance import EllipticEnvelope


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['elliptic_envelope']
    cls = EllipticEnvelope(
        contamination=params['contamination'],
        random_state=params['random_seed']
        )
    detect_outliers(cls)
