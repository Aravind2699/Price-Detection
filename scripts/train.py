import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import datetime
from scripts.data_loader import clean_chunk, add_features, data_load_and_preprocess
from scripts.model import build_pipeline

# === 4. MAIN TRAINING SCRIPT ===--
def process_and_train(csv_path):
    data = data_load_and_preprocess(csv_path)
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
    dump(pipeline, 'git/models/taxi_model.joblib')
    print("Model saved to git/models/taxi_model.joblib")

# === 5. PREDICT NEW DATA ===
def predict_new_data(new_data_path):
    model = load('git/models/taxi_model.joblib')
    new_data = pd.read_csv(new_data_path)
    new_data = add_features(new_data)
    X_new = new_data[['trip_distance', 'passenger_count', 'hour', 'dayofweek']]
    predictions = model.predict(X_new)
    new_data['predicted_fare'] = predictions
    new_data.to_csv('models/predicted_fares.csv', index=False)
    print("Predictions saved to models/predicted_fares.csv")

# === 6. PREDICT SINGLE SAMPLE ===++++--
def predict_single_sample(trip_distance, passenger_count, pickup_datetime):
    print('predicting')
    model = load('git/models/taxi_model.joblib')
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
    # process_and_train('Drives/data/raw/2023_Yellow_Taxi_Trip_Data.csv')
    # predict_new_data('Drives/data/raw/new_trip_data.csv')
    predict_single_sample(6.5, 2, datetime.datetime.now())