from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define global @tf.function to prevent retracing
@tf.function(reduce_retracing=True)
def predict_with_model(model, input_tensor):
    return model(input_tensor, training=False)

# Load model and scaler
def load_model_and_scaler(stock_name):
    model_filename = os.path.join(MODEL_DIR, f'{stock_name}_model.pkl')
    scaler_filename = os.path.join(MODEL_DIR, f'{stock_name}_scaler.pkl')

    # Fix: Use correct variable names
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        with open(model_filename, "rb") as model_file:
            model = pickle.load(model_file)
        with open(scaler_filename, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler

    return None, None


# Preprocess stock data
def preprocess_data(stock_name):
    latest_data = yf.download(f"{stock_name}.NS", period="10d", interval="1d")
    
    if latest_data.empty or len(latest_data) < 2:
        return None, "Not enough data for prediction."

    latest_data['10_EMA'] = latest_data['Close'].ewm(span=10, adjust=False).mean()
    latest_data['50_EMA'] = latest_data['Close'].ewm(span=50, adjust=False).mean()
    latest_data['200_EMA'] = latest_data['Close'].ewm(span=200, adjust=False).mean()
    latest_data['Open-Close'] = latest_data['Open'] - latest_data['Close']
    latest_data['Low-High'] = latest_data['High'] - latest_data['Low']

    feature_columns = ["High", "Low", "Open", "Close", "Open-Close", "Low-High", "Volume", "10_EMA", "50_EMA", "200_EMA"]
    latest_day = latest_data.iloc[-2]  # Use the previous day's data

    features = np.array([
        latest_day["High"], latest_day["Low"], latest_day["Open"], latest_day["Close"],
        latest_day["Open-Close"], latest_day["Low-High"], latest_day["Volume"],
        latest_day["10_EMA"], latest_day["50_EMA"], latest_day["200_EMA"]
    ], dtype=np.float32).reshape(1, -1)
    print(features)
    return features, None

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        stock_name = data.get('stock_name')

        if not stock_name:
            return jsonify({"error": "Stock name is required"}), 400

        # Load model & scaler
        model, scaler = load_model_and_scaler(stock_name)
        if model is None or scaler is None:
            return jsonify({"error": f"Model or scaler not found for {stock_name}"}), 404

        # Get features from previous day's data
        features, error = preprocess_data(stock_name)
        if features is None:
            return jsonify({"error": error}), 500

        # Scale input data
        scaled_features = scaler.transform(features)
        scaled_features = scaled_features.reshape(1, 1, -1)  # Shape: (1,1,10)

        # Convert to tensor to prevent retracing
        input_tensor = tf.convert_to_tensor(scaled_features, dtype=tf.float32)

        # Predict
        next_day_scaled = predict_with_model(model, input_tensor).numpy()

        # Prepare for inverse transformation
        dummy_array = np.zeros((1, 10), dtype=np.float32)
        dummy_array[:, 3] = next_day_scaled  # Insert predicted 'Close' price

        # Convert back to original scale
        predicted_price = scaler.inverse_transform(dummy_array)[:, 3][0]

        return jsonify({"stock": stock_name, "predicted_price": round(float(predicted_price), 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4960)
