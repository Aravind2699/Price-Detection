import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def clean_chunk(df):
    df = df.dropna(subset=['fare_amount', 'trip_distance', 'passenger_count'])
    df = df[(df['trip_distance'] > 0) & 
        (df['fare_amount'] > 0) & 
        (df['passenger_count'] > 0)]
    return df

# === 2. FEATURE ENGINEERING ===
def add_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    return df

# === 3. DATA Visualization--
def data_visualization(chunk):
        chunk=clean_chunk(chunk)
        chunk=add_features(chunk)

        #correlation444444
        # sns.heatmap(chunk.corr(), annot=True, cmap='coolwarm')
        # plt.show()

        #direc comparison for two colmns
        # sns.boxplot(x='RatecodeID', y='fare_amount', data=chunk)
        # plt.show()

        #using model to predict importance of column
        from sklearn.ensemble import RandomForestRegressor
        print(chunk.head())
        print(chunk[['total_amount', 'fare_amount']].head())

        X = chunk[['VendorID', 'hour', 'dayofweek',
       'passenger_count', 'trip_distance', 'RatecodeID',
       'PULocationID', 'DOLocationID', 
       'airport_fee']]
        y = chunk['fare_amount']

        model = RandomForestRegressor().fit(X, y)
        importances = model.feature_importances_
        sns.barplot(x=X.columns, y=importances)
        plt.show()

# === 4. DATA LOADING AND PREPROCESSING ===--
def data_load_and_preprocess(csv_path):
    chunk_iter = pd.read_csv(csv_path, chunksize=100000)
    all_chunks = []
    for i, chunk in enumerate(chunk_iter):
        # data_visualization(chunk)
        print(f"Processing chunk {i+1}...")
        chunk = clean_chunk(chunk)
        chunk = add_features(chunk)
        print(max( chunk["trip_distance"]))
        all_chunks.append(chunk[['trip_distance', 'passenger_count', 'hour', 'dayofweek', 'fare_amount']])
        if i==1 :
            break
    data = pd.concat(all_chunks)
    print(data.shape)
    return data

data_load_and_preprocess('Drives/data/raw/2023_Yellow_Taxi_Trip_Data.csv')
