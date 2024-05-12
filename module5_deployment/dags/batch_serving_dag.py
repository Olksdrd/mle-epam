from datetime import datetime

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from dagtasks.serving import (get_batch,
                              # check_columns,
                              check_for_outliers,
                              get_predictions
                              )

import os
# Set up working directory in module5 folder
module5_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(module5_path)

with DAG(
    dag_id='batch_serving',
    start_date=datetime(2024, 4, 20),
    schedule='*/5 * * * *',
    catchup=False,
    tags=['module5'],
) as dag:

    wait_for_model = FileSensor(
        task_id="wait_for_model",
        filepath=module5_path + "/models/hgb/model.pkl",
        poke_interval=30,
        retries=1
    )

    test_model = BashOperator(
        task_id='model_tests',
        bash_command=f'python3 {module5_path}/tests/model_tests.py'
    )

    create_batch = PythonOperator(
        task_id='get_new_batch',
        python_callable=get_batch,
        op_kwargs={'batch_size': 20},
    )

    # signature_check = PythonOperator(
    #     task_id='signature_check',
    #     python_callable=check_columns
    # )

    test_batch = BashOperator(
        task_id='batch_tests',
        bash_command=f'python3 {module5_path}/tests/batch_tests.py'
    )

    outliers = PythonOperator(
        task_id='check_for_outliers',
        python_callable=check_for_outliers
    )

    generate_predictions = PythonOperator(
        task_id='get_predictions',
        python_callable=get_predictions
    )

    wait_for_model >> test_model >> create_batch
    create_batch >> test_batch >> outliers >> generate_predictions
