# Real-Time Fraud Detection System 🚨

## Overview
This project implements a **Real-Time Fraud Detection System** using machine learning models (Isolation Forest & Autoencoder). The system detects fraudulent transactions in real-time, visualizes anomalies on a live dashboard, and sends email alerts upon detection.

## Features
- **Real-Time Fraud Detection**: Uses Isolation Forest and Autoencoder models.
- **Live Dashboard**: Interactive dashboard displaying detected frauds in real-time.
- **Email Alerts**: Sends email notifications whenever fraud is detected.
- **Data Source**: The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

## Tech Stack
- **Machine Learning**: Isolation Forest, Autoencoders (Keras)
- **Web App**: Streamlit for dashboard
- **Email Alerts**: smtplib (Gmail) for notifications
- **Data**: Credit Card Fraud Dataset (CSV)
- **Python Libraries**: pandas, scikit-learn, Keras, Streamlit, smtplib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/real-time-fraud-detection.git
2. Navigate to the project folder:
    cd real-time-fraud-detection
3. Set up a virtual environment (recommended):
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
4. Install the required dependencies:
       pip install -r requirements.txt
5. Usage:
    i)Run the simulation: python realtime_simulation.py
    ii)Launch the dashboard:streamlit run dashboard_app.py
    iii)Check your email: If fraud is detected, you'll receive an email alert.
6. License
    This project is licensed under the MIT License - see the LICENSE file for details.    

       




