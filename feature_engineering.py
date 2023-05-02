#The code calculates the moving average of the trading volume of 30 days per each stock and ETF, and stores it in a newly added column called 'vol_moving_avg'. It also calculates the rolling median and stores it in a newly added column called 'adj_close_rolling_med':

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from multiprocessing import Pool
import numpy as np


def process_chunk(chunk):

    chunk['vol_moving_avg'] = chunk['Volume'].rolling(window=30).mean()
    chunk['adj_close_rolling_med'] = chunk['Adj Close'].rolling(window=30).median()

    return chunk

def feature_engineering():

    # Load the processed data into a DataFrame
    processed_df = pd.read_csv("processed_files/processed_data.csv")

    # Split the DataFrame into chunks
    chunks = np.array_split(processed_df, 4)

    # Creating pool of workers and creating four chunks:
    with Pool(processes=4) as pool:
        processed_chunks = pool.map(process_chunk, chunks)

    # Concatenate the processed chunks back into a DataFrame
    processed_df = pd.concat(processed_chunks)

    if not os.path.exists('processed_parquet_file'):
        os.mkdir('processed_parquet_file')

    processed_df.to_parquet("processed_parquet_file/processed_data.parquet")
    print("DONE")
