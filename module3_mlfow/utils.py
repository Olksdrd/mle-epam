import tempfile
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import estimator_html_repr
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error
                             )


import mlflow
from mlflow.models.signature import infer_signature



def eval_model(model, X_train, X_test, y_train, y_test):

  train_predicted = model.predict(X_train)
  train_r2 = r2_score(y_train, train_predicted)
  train_rmse = mean_squared_error(y_train, train_predicted, squared=False)
  train_mae = mean_absolute_error(y_train, train_predicted)

  test_predicted = model.predict(X_test)
  test_r2 = r2_score(y_test, test_predicted)
  test_rmse = mean_squared_error(y_test,test_predicted, squared=False)
  test_mae = mean_absolute_error(y_test, test_predicted)

#   print(f'R2  : {r2:.2f}')
#   print(f'RMSE: {rmse:.2f}')
#   print(f'MAE : {mae:.2f}')

  return {'train_r2': train_r2, 'train_rmse': train_rmse, 'train_mae': train_mae,
          'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mae': test_mae}



num_pipe_v1 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])



preprocessor_v1 = ColumnTransformer([
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object')),
    ('num', num_pipe_v1, make_column_selector(dtype_include=np.number))
], verbose_feature_names_out=False)


def default_mlflow_run(regressor,
                       run_name,# exp_id,
                       X_train, X_test, y_train, y_test):

    with mlflow.start_run(
        run_name=run_name,# experiment_id=exp_id
        ) as run:


        pipe = Pipeline([
            ('preprocessing', preprocessor_v1),
            ('regressor', regressor)
        ])

        model_name = type(pipe['regressor']).__name__

        print(f'Logging {model_name} model...')
        print('Run ID: ', run.info.run_id)

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
                                registered_model_name=model_name)
