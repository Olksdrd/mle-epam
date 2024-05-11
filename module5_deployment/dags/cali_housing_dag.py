import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from dagtasks.loading import load_data, read_data
from dagtasks.cleaning import (merge_tables,
                               str_to_num,
                               uniform_missing_indicator,
                               rename_unknown_category,
                               )
from dagtasks.preprocessing import (feature_interactions,
                                    train_val_split)
from dagtasks.training import detect_outliers, train_model, validate_model


# Set up working directory in 'module4_airflow' folder
module4_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(module4_path)


with DAG(
    dag_id='housing_hgb',
    description='HGB prediction for California housing dataset.',
    start_date=datetime(2024, 4, 20),
    schedule=None,
    catchup=False,
    tags=['module5', 'cali_housing']
) as dag:

    with TaskGroup('load', tooltip='Loading dataset') as load_group:

        get_dataset = PythonOperator(
            task_id='load_data',
            python_callable=load_data
        )

        read_dataset = PythonOperator(
            task_id='read_data',
            python_callable=read_data
        )

        get_dataset >> read_dataset

    with TaskGroup('clean', tooltip='Data cleaning') as clean_group:

        convert_dtypes = PythonOperator(
            task_id='convert_dtypes',
            python_callable=str_to_num
        )

        missing_indicator = PythonOperator(
            task_id='fix_NAs_num',
            python_callable=uniform_missing_indicator
        )

        unknown_category = PythonOperator(
            task_id='fix_NAs_category',
            python_callable=rename_unknown_category
        )

        perform_join = PythonOperator(
            task_id='table_join',
            python_callable=merge_tables
        )

        # columns_signature = PythonOperator(
        #     task_id='save_columns_signature',
        #     python_callable=save_columns_signature
        # )

        convert_dtypes >> missing_indicator
        [missing_indicator, unknown_category] >> perform_join
        # perform_join >> columns_signature

    with TaskGroup(group_id='preprocess',
                   tooltip='Data preprocessing') as preprocess_group:

        generate_features = PythonOperator(
            task_id='feature_interactions',
            python_callable=feature_interactions
        )

        data_split = PythonOperator(
            task_id='train_val_split',
            python_callable=train_val_split
        )

        generate_features >> data_split

    with TaskGroup(group_id='train',
                   tooltip='Model training and evaluation') as train_group:

        outliers = PythonOperator(
            task_id='detect_outliers',
            python_callable=detect_outliers
        )

        training = PythonOperator(
            task_id='train_model',
            python_callable=train_model
        )

        cross_validation = PythonOperator(
            task_id='validate_model',
            python_callable=validate_model
        )

        outliers >> training >> cross_validation

    load_group >> clean_group >> preprocess_group >> train_group
