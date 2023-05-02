import pandas as pd
import numpy as np
import os
import zipfile
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import multiprocessing
from multiprocessing import Pool
import pyarrow as pa

def integrate_ml_training():

    df_sample = pd.read_parquet('processed_parquet_file/processed_data.parquet')
    df = df_sample.sample(frac=1.0, random_state=42)

    # Convert Date column to datetime type and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Remove rows with NaN values
    df.dropna(inplace=True)

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    X = df[features]
    y = df[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a GradientBoostingRegressor model on the data
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if not os.path.exists('models'):
        os.mkdir('models')
        
    model_path = 'models/gradient_boosting.joblib'
    joblib.dump(model, model_path)

    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Set up logging to console and file
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Log to file
    file_handler = logging.FileHandler('logs/training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log training metrics
    logger.info(f'Training completed. Mean Absolute Error: {mae}. Mean Squared Error: {mse}.')



if __name__ == '__main__':
    # Create a multiprocessing pool with 4 processes
    pool = multiprocessing.Pool(4)

    # Call the integrate_ml_training function with each process
    results = []
    for i in range(4):
        results.append(pool.apply_async(integrate_ml_training))

    # Wait for all processes to finish and collect any errors
    errors = []
    for result in results:
        try:
            result.get()
        except Exception as e:
            errors.append(e)

    # Print any errors that occurred
    if errors:
        print(f"{len(errors)} errors occurred during training:")
        for e in errors:
            print(str(e))
    else:
        print("Training completed successfully.")
