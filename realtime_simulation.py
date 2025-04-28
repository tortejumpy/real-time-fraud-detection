import pandas as pd
import numpy as np
import time
import joblib

# Load saved model and scaler
model = joblib.load('models/isolation_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load data
df = pd.read_csv('data/creditcard.csv')

# Drop label
X = df.drop('Class', axis=1)
y = df['Class']

# Simulate real-time transactions
for i in range(100):  # Simulate 100 transactions
    # Randomly pick one transaction
    random_index = np.random.randint(0, len(X))
    transaction = X.iloc[random_index]

    # Scale it
    transaction_scaled = scaler.transform([transaction])

    # Predict
    pred = model.predict(transaction_scaled)

    if pred[0] == -1:
        print(f"🚨 Fraud Alert Detected at index {random_index}!")
        print(transaction)
        print("-" * 50)
    else:
        print(f"✅ Legit Transaction at index {random_index}")

    # Wait for a short time to simulate real-time
    time.sleep(0.5)  # 0.5 seconds delay
