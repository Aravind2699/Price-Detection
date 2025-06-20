import pandas as pd
import numpy as np

def clean_chunk(df):
    df = df.dropna(subset=['fare_amount', 'trip_distance', 'passenger_count'])
    df = df[(df['fare_amount'] > 0) & (df['trip_distance'] > 0)]
    return df

# === 2. FEATURE ENGINEERING ===
def add_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    return df

# === 4. DATA LOADING AND PREPROCESSING ===
def data_load_and_preprocess(csv_path):
    chunk_iter = pd.read_csv(csv_path, chunksize=100000)
    all_chunks = []
    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i+1}...")
        chunk = clean_chunk(chunk)
        chunk = add_features(chunk)
        all_chunks.append(chunk[['trip_distance', 'passenger_count', 'hour', 'dayofweek', 'fare_amount']])
        if i==1 :
            break
    data = pd.concat(all_chunks)
    print("Min:", data["fare_amount"].min())
    print("Max:", data["fare_amount"].max())
    print(data.shape)
    return data