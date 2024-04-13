import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer


from sklearn.utils import estimator_html_repr
from sklearn.metrics import (r2_score, PredictionErrorDisplay,
                             mean_absolute_error,mean_squared_error)

import mlflow
from mlflow.models.signature import infer_signature



num_pipe_v1 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

preprocessor_v1 = ColumnTransformer([
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object')),
    ('num', num_pipe_v1, make_column_selector(dtype_include=np.number))
], verbose_feature_names_out=False)


class FeatureInteractions(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    X['lat*long'] = X['latitude'] * X['longitude']
    X['lat/long'] = X['latitude'] / X['longitude']
    X['rooms_per_house'] = X['total_rooms'] / X['households']
    X['bedrooms_per_house'] = X['total_bedrooms'] / X['households']
    X['rooms_per_pop'] = X['total_rooms'] / X['population']
    X['bedrooms_per_pop'] = X['total_bedrooms'] / X['population']
    X['bedroom_frac'] = X['total_bedrooms'] / X['total_rooms']
    X['pop_per_house'] = X['population'] / X['households']

    X['rooms_per_income'] = X['total_rooms'] / X['median_income']
    X['bedrooms_per_income'] = X['total_bedrooms'] / X['median_income']

    return X



num_union = FeatureUnion([
    ('identity', 'passthrough'),
    ('binning', make_pipeline(KBinsDiscretizer(strategy='kmeans'),
                              VarianceThreshold(threshold=0.1)))
])

num_pipe_v2 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('union', num_union)
])

preprocessor_v2 = ColumnTransformer([
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object')),
    ('num', num_pipe_v2, make_column_selector(dtype_include=np.number))
], verbose_feature_names_out=False)




def load_data(path, random_state=42):
    X = pd.read_csv(path)
    y = X.pop('median_house_value')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=random_state)
    
    return X_train, X_test, y_train, y_test



def eval_model(model, X_train, X_test, y_train, y_test):

  train_predicted = model.predict(X_train)
  train_r2 = r2_score(y_train, train_predicted)
  train_rmse = mean_squared_error(y_train, train_predicted, squared=False)
  train_mae = mean_absolute_error(y_train, train_predicted)

  test_predicted = model.predict(X_test)
  test_r2 = r2_score(y_test, test_predicted)
  test_rmse = mean_squared_error(y_test,test_predicted, squared=False)
  test_mae = mean_absolute_error(y_test, test_predicted)

  return {'train_r2': train_r2, 'train_rmse': train_rmse, 'train_mae': train_mae,
          'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mae': test_mae}



def default_mlflow_run(regressor, X_train, X_test, y_train, y_test):

    with mlflow.start_run() as run:

        pipe = Pipeline([
            ('preprocessing', preprocessor_v1),
            ('regressor', regressor)
        ])

        model_name = type(pipe['regressor']).__name__

        print(f'Logging {model_name} model...')
        # print('Run ID: ', run.info.run_id)

        pipe.fit(X_train, y_train)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = eval_model(pipe, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        res_plot = plt.figure()
        display = PredictionErrorDisplay.from_predictions(y_test, pipe.predict(X_test), ax=plt.gca())
        plt.title('Residuals Plot')
        mlflow.log_figure(res_plot, 'residual_plot.png')

        estimator = estimator_html_repr(pipe)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, 'my_estimator.html')
            path.write_text(estimator, encoding="utf-16")

            mlflow.log_artifact(path)

        model_signature = infer_signature(X_train, y_train,
                                        # params={'data_version': '1.0'}
                                        )
        
        mlflow.sklearn.log_model(pipe,
                                artifact_path=model_name,
                                signature=model_signature,
                                registered_model_name=model_name,
                                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        



def pipe_ver2_mlflow_run(regressor, X_train, X_test, y_train, y_test):

    with mlflow.start_run() as run:

        pipe = Pipeline([
        ('interactions', FeatureInteractions()),
        ('preprocessing', preprocessor_v2),
        ('regressor', regressor)
    ])

        model_name = type(pipe['regressor']).__name__

        print(f'Logging {model_name} model...')
        # print('Run ID: ', run.info.run_id)

        pipe.fit(X_train, y_train)

        mlflow.log_params(pipe['regressor'].get_params())

        metrics = eval_model(pipe, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        res_plot = plt.figure()
        display = PredictionErrorDisplay.from_predictions(y_test, pipe.predict(X_test), ax=plt.gca())
        plt.title('Residuals Plot')
        mlflow.log_figure(res_plot, 'residual_plot.png')

        estimator = estimator_html_repr(pipe)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, 'my_estimator.html')
            path.write_text(estimator, encoding="utf-16")

            mlflow.log_artifact(path)

        model_signature = infer_signature(X_train, y_train,
                                        # params={'data_version': '1.0'}
                                        )
        
        mlflow.sklearn.log_model(pipe,
                                artifact_path=model_name,
                                signature=model_signature,
                                registered_model_name=model_name,
                                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)