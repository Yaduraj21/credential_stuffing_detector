import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def generate_advanced_dataset(n_samples=2000):
    # --- 1. NORMAL HUMAN BEHAVIOR ---
    normal = pd.DataFrame({
        'attempts_per_min': np.random.poisson(2, n_samples),
        'fail_ratio': np.random.uniform(0.0, 0.3, n_samples),     # Humans fail sometimes
        'unique_accounts': np.random.randint(1, 2, n_samples),    # Humans usually use 1 account
        'device_change': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]), # Rare device change
        'geo_anomaly': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]), # Rare travel
        'honeypot': np.zeros(n_samples),                          # Humans don't see/fill hidden fields
        'typing_speed': np.random.uniform(2.0, 5.0, n_samples)    # 2-5 seconds to type
    })

    # --- 2. CREDENTIAL STUFFING (BOT) BEHAVIOR ---
    bots = pd.DataFrame({
        'attempts_per_min': np.random.poisson(60, n_samples // 10),
        'fail_ratio': np.random.uniform(0.8, 1.0, n_samples // 10), # High failure rate
        'unique_accounts': np.random.randint(10, 100, n_samples // 10), # Testing many users
        'device_change': np.ones(n_samples // 10),                 # Often spoofing new devices
        'geo_anomaly': np.random.choice([0, 1], n_samples // 10, p=[0.5, 0.5]),
        'honeypot': np.random.choice([0, 1], n_samples // 10, p=[0.2, 0.8]), # Bots often trip honeypots
        'typing_speed': np.random.uniform(0.1, 0.5, n_samples // 10) # Very fast/robotic
    })

    df = pd.concat([normal, bots]).sample(frac=1).reset_index(drop=True)
    return df

# Initialize folders and build model
os.makedirs('ml_engine/models', exist_ok=True)
df = generate_advanced_dataset()
df.to_csv('ml_engine/data/training_data_v2.csv', index=False)

# Train the model [cite: 49]
# We set contamination to 0.09 because bots are ~9% of our generated data
model = IsolationForest(n_estimators=100, contamination=0.09, random_state=42)
model.fit(df)

joblib.dump(model, 'ml_engine/models/detector_v1.pkl')
print("Advanced ML Model Trained with 7 Features!")