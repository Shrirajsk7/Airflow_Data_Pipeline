# Ingesting and processing raw stock market datasets:

import pandas as pd
import numpy as np
import os
import multiprocessing


def process_csv_file(file_path):

    df = pd.read_csv(file_path)

    # Select the required columns from the dataframe
    processed_df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    return processed_df

def raw_data_processing():

    etf_files = [os.path.join("./etfs", f) for f in os.listdir("./etfs") if f.endswith('.csv')][:500]
    stock_files = [os.path.join("./stocks", f) for f in os.listdir("./stocks") if f.endswith('.csv')][:500]
    file_paths = etf_files + stock_files

    #Making use of all the available processor in the system:
    pool = multiprocessing.Pool()
    results = pool.map(process_csv_file, file_paths)

    #merging all the data from etf and stocks folder:
    merged_df = pd.concat(results)

    symbol_df = pd.read_csv("symbols_valid_meta.csv")

    # Merge the symbol_df dataframe with the merged dataframe to add the Symbol and Security Name columns:
    merged_df = pd.merge(merged_df, symbol_df[['Symbol', 'Security Name']], left_index=True, right_index=True)

    # Saving the merged dataframe to a CSV file in the 'processed_file' folder:
    if not os.path.exists("processed_files"):
        os.makedirs("processed_files")
    
    #Saving the CSV format of the merged file:
    merged_df.to_csv("processed_files/processed_data.csv", index=False)

    #Saving the parquet format of the merged file:
    merged_df.to_parquet("processed_files/processed_data.parquet", index=False)
    
    return "processed_files/processed_data.csv"
