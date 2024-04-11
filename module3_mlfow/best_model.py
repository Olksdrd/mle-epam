
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from utils import eval_model, FeatureInteractions, BinFeatures, preprocessor_trees

import mlflow
from mlflow.models.signature import infer_signature

#mlflow.set_tracking_uri("http://0.0.0.0:5001/")
mlflow.set_tracking_uri("http://172.17.0.2:5000/")

RANDOM_SEED = 42

# import data
X = pd.read_csv('housing.csv')
y = X.pop('median_house_value')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
                                                    random_state = 42)

# Create experiment

try:
    experiment_description = (
        'Trying basic models with default parameters and no feature engineering.'
    )

    experiment_tags = {
        'env': 'dev',
        'project_name': 'California_housing',
        'mlflow.note.content': experiment_description,
        'pipeline_version': '1.0'
    }

    exp_id = mlflow.create_experiment(
        name='Best_Model',
        tags=experiment_tags
    )
except:
    pass

exp_id = mlflow.get_experiment_by_name('Best_Model').experiment_id

# Run experiments


def main(regressor, run_name,
         X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:


        pipe = Pipeline([
        ('interactions', FeatureInteractions()),
        ('binning', BinFeatures(['population', 'median_income'])),
        ('preprocessing', preprocessor_trees),
        ('regressor', regressor)
    ])

        model_name = type(pipe['regressor']).__name__

        print(f'Logging {model_name} model...')
        print('Run ID: ', run.info.run_id)

        pipe.fit(X_train, y_train)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = eval_model(pipe, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        model_signature = infer_signature(X_train, y_train,
                                        # params={'data_version': '1.0'}
                                        )
        
        mlflow.sklearn.log_model(pipe,
                                artifact_path=model_name,
                                signature=model_signature,
                                registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

if __name__ == '__main__':
    main(HistGradientBoostingRegressor(random_state=RANDOM_SEED), 'hgb_best',
         X_train_raw, X_test_raw, y_train, y_test)