import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tensorflow.keras.models import load_model
import threading
import matplotlib.pyplot as plt

# Set up page configuration for light/dark theme toggle
st.set_page_config(page_title="🚨 Real-Time Fraud Detection", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------
# Email Alert Function (Asynchronous)
# ------------------------------------------
def send_email_alert(transaction_details):
    sender_email = "your_email@gmail.com"          # 🔥 Your email
    receiver_email = "receiver_email@gmail.com"     # 🔥 Receiver email
    password = "your_app_password"                  # 🔥 Gmail app-specific password

    subject = "🚨 Fraudulent Transaction Detected!"
    body = f"A fraudulent transaction was detected:\n\n{transaction_details}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Alert Email Sent!")
    except Exception as e:
        print(f"Error sending email: {e}")

# ------------------------------------------
# Load Models
# ------------------------------------------
iso_model = joblib.load('models/isolation_forest_model.pkl')
iso_scaler = joblib.load('models/scaler.pkl')
autoencoder_model = load_model('models/autoencoder_model.h5', compile=False)
auto_scaler = joblib.load('models/autoencoder_scaler.pkl')

# ------------------------------------------
# Load and Prepare Data
# ------------------------------------------
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Scale data
X_iso = iso_scaler.transform(X)
X_auto = auto_scaler.transform(X)

# ------------------------------------------
# Streamlit App
# ------------------------------------------

# Title Section
st.markdown("<h1 class='title'>🚨 Real-Time Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

# Theme Section (Light/Dark Theme Toggle)
theme_option = st.selectbox("Choose Theme", ["Light", "Dark"])

if theme_option == "Light":
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f9;
                color: #2C3E50;
            }
            .title {
                font-size: 36px;
                color: #2C3E50;
                font-weight: 600;
                text-align: center;
            }
            .metric-container {
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-top: 30px;
            }
            .metric-container .stMetric {
                background-color: #34495E;
                color: white;
                border-radius: 10px;
                padding: 20px;
                font-size: 18px;
            }
            .fraud-flag {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
                border-radius: 15px;
                margin-top: 15px;
            }
            .fraud {
                background: linear-gradient(90deg, #FF5E5B, #FF3D3D);
                color: white;
            }
            .normal {
                background: linear-gradient(90deg, #2ECC71, #27AE60);
                color: white;
            }
            .transaction-details {
                font-size: 16px;
                margin-top: 10px;
                padding: 10px;
                background-color: #f1f1f1;
                border-radius: 10px;
            }
            .footer {
                text-align: center;
                padding: 20px;
                background-color: #2C3E50;
                color: white;
                position: fixed;
                width: 100%;
                bottom: 0;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body {
                background-color: #2C3E50;
                color: white;
            }
            .title {
                font-size: 36px;
                color: white;
                font-weight: 600;
                text-align: center;
            }
            .metric-container {
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-top: 30px;
            }
            .metric-container .stMetric {
                background-color: #34495E;
                color: white;
                border-radius: 10px;
                padding: 20px;
                font-size: 18px;
            }
            .fraud-flag {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
                border-radius: 15px;
                margin-top: 15px;
            }
            .fraud {
                background: linear-gradient(90deg, #FF5E5B, #FF3D3D);
                color: white;
            }
            .normal {
                background: linear-gradient(90deg, #2ECC71, #27AE60);
                color: white;
            }
            .transaction-details {
                font-size: 16px;
                margin-top: 10px;
                padding: 10px;
                background-color: #f1f1f1;
                border-radius: 10px;
            }
            .footer {
                text-align: center;
                padding: 20px;
                background-color: #2C3E50;
                color: white;
                position: fixed;
                width: 100%;
                bottom: 0;
            }
        </style>
    """, unsafe_allow_html=True)

# ------------------------------------------
# Statistics Section
# ------------------------------------------
total_transactions = 0
fraud_detected = 0

placeholder = st.empty()

# Initialize lists for plotting
normal_count = 0
fraud_count = 0

# Process transactions in a batch
batch_size = 50  # Define the batch size to process
for i in range(0, len(X), batch_size):
    batch_transactions = X.iloc[i:i+batch_size]
    batch_transactions_scaled_iso = X_iso[i:i+batch_size]
    batch_transactions_scaled_auto = X_auto[i:i+batch_size]

    # Predict using Isolation Forest and Autoencoder models
    iso_preds = iso_model.predict(batch_transactions_scaled_iso)
    recon = autoencoder_model.predict(batch_transactions_scaled_auto)
    mse = np.mean(np.power(batch_transactions_scaled_auto - recon, 2), axis=1)
    auto_preds = [1 if m > np.percentile(mse, 95) else -1 for m in mse]

    # Combine predictions and detect fraud
    for idx, (iso_pred, auto_pred) in enumerate(zip(iso_preds, auto_preds)):
        if iso_pred == -1 or auto_pred == 1:
            fraud_flag = "🚨 FRAUD DETECTED"
            color_class = "fraud"
            fraud_detected += 1
            fraud_count += 1

            # 👉 Send email alert asynchronously
            transaction_details = batch_transactions.iloc[idx].to_dict()  # Extract transaction details for alert
            threading.Thread(target=send_email_alert, args=(transaction_details,)).start()

        else:
            fraud_flag = "Normal Transaction"
            color_class = "normal"
            normal_count += 1

        total_transactions += 1

        # Update Dashboard with stylish layout
        with placeholder.container():
            st.markdown(f"<div class='metric-container'><div class='stMetric'><h3>Total Transactions</h3><p>{total_transactions}</p></div>"
                        f"<div class='stMetric'><h3>Frauds Detected</h3><p>{fraud_detected}</p></div></div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='transaction-details'><b>Transaction {total_transactions}:</b> {batch_transactions.iloc[idx]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='fraud-flag {color_class}'>{fraud_flag}</div>", unsafe_allow_html=True)

    time.sleep(1)  # Simulate live data fetching delay

# Bar chart for fraud vs normal transactions
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Normal', 'Fraud'], [normal_count, fraud_count], color=['#2ECC71', '#FF3D3D'])
ax.set_title('Fraud vs Normal Transactions')
ax.set_ylabel('Count')
st.pyplot(fig)

# Footer
st.markdown("<div class='footer'>Powered by Streamlit | Fraud Detection System</div>", unsafe_allow_html=True)
