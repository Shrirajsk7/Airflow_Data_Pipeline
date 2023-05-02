from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from extract_data import extract_data
from raw_data_processing import raw_data_processing
from feature_engineering import feature_engineering
from ml_training import integrate_ml_training
from datetime import datetime


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020, 11, 8),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}


dag = DAG(
    'stock_dag',
    default_args=default_args,
    description='Our first DAG with ETL process!',
    schedule_interval=None,
)


extract_data = PythonOperator(
    task_id='Data_Extraction',
    python_callable=extract_data,
    dag=dag,
)

raw_data_processing = PythonOperator(
    task_id='process_raw_data',
    python_callable=raw_data_processing,
    dag=dag,
)

feature_engineering = PythonOperator(
    task_id='perform_feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

integrate_ml_training = PythonOperator(
    task_id='integrate_ml_training',
    python_callable=integrate_ml_training,
    dag=dag,
)


extract_data >> raw_data_processing >>feature_engineering >> integrate_ml_training


