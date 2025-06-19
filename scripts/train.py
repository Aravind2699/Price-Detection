import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump, load
import os
from sklearn.neural_network import MLPRegressor

# === 1. CLEANING FUNCTION ===
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

# === 3. PREPROCESSING PIPELINE ===
def build_pipeline():
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    #   ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    return pipeline

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
# === 4. MAIN TRAINING SCRIPT ===
def process_and_train(csv_path):
    data=data_load_and_preprocess(csv_path)
    print('training')
    X = data[['trip_distance', 'passenger_count', 'hour', 'dayofweek']]
    y = data['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    os.makedirs('models', exist_ok=True)
    dump(pipeline, 'models/taxi_model_neural.joblib')
    print("Model saved to model_output/taxi_model_neural.joblib")

# === 5. PREDICT NEW DATA ===
def predict_new_data(new_data_path):
    model = load('model_output/taxi_model_neural.joblib')
    new_data = pd.read_csv(new_data_path)
    new_data = add_features(new_data)
    X_new = new_data[['trip_distance', 'passenger_count', 'hour', 'dayofweek']]
    predictions = model.predict(X_new)
    new_data['predicted_fare'] = predictions
    new_data.to_csv('model_output/predicted_fares.csv', index=False)
    print("Predictions saved to model_output/predicted_fares.csv")

# === 6. PREDICT SINGLE SAMPLE ===
def predict_single_sample(trip_distance, passenger_count, pickup_datetime):
    print('predicting')
    model = load('models/taxi_model_neural.joblib')
    dt = pd.to_datetime(pickup_datetime)
    hour = dt.hour
    dayofweek = dt.dayofweek
    X_new = pd.DataFrame([[trip_distance, passenger_count, hour, dayofweek]], 
                         columns=['trip_distance', 'passenger_count', 'hour', 'dayofweek'])
    predicted_fare = model.predict(X_new)[0]
    print(f"Predicted Fare: ${predicted_fare:.2f}")
    
    return predicted_fare

# === 7. RUN ===
if __name__ == '__main__':
    process_and_train('/home/aravind/aravind/ML_product/datasets/2023_Yellow_Taxi_Trip_Data.csv')  # You can replace with full dataset
    # predict_new_data('new_trip_data.csv')
    predict_single_sample(2.5, 1, '2023-05-01 18:30:00')
    # data_load_and_preprocess('/home/aravind/aravind/ML_product/datasets/2023_Yellow_Taxi_Trip_Data.csv')