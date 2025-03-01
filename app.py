from flask import Flask, request, jsonify
import pickle
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Initialize Flask app
print(pd.__version__)
print(sklearn.__version__)
print(pickle.format_version)
print(tf.__version__)
app = Flask(__name__)

# Load the model and scaler for a specific stock
def load_model_and_scaler(stock_name):
    try:
        model_filename = f'{stock_name}_model.pkl'
        scaler_filename = f'{stock_name}_scaler.pkl'

        # Load the model and scaler using pickle
        model = pickle.load(open(model_filename, 'rb'))
        scaler = pickle.load(open(scaler_filename, 'rb'))

        return model, scaler
    except FileNotFoundError:
        return None, None

# Prepare and preprocess data for the stock
def preprocess_data(stock_name):
    try:
        # Load the CSV file for the stock (this assumes that the stock data is in the current directory)
        data = pd.read_csv(f'{stock_name}.csv', header=None, skiprows=2)
        # Assign proper column names
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Clean and preprocess data
        data = data.drop(index=0)
        data.reset_index(drop=True, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = data['Date'].dt.day
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year
        data['QuarterEnd'] = np.where(data['Month'] % 3 == 0, 1, 0)

        # Feature Engineering: Using only necessary features
        data['Open-Close'] = data['Open'] - data['Close']
        data['Low-High'] = data['High'] - data['Low']
        data['10_EMA'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['50_EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['200_EMA'] = data['Close'].ewm(span=200, adjust=False).mean()

        # Normalize Features (use only relevant columns)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['High', 'Low', 'Open', 'Close', 'Open-Close', 'Low-High', 'Volume', '10_EMA', '50_EMA', '200_EMA']])

        return scaled_data, scaler
    except Exception as e:
        return None, str(e)

# API endpoint to get the predicted closing price
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        data = request.get_json()
        stock_name = data['stock_name']

        # Load the model and scaler for the requested stock
        model, scaler = load_model_and_scaler(stock_name)

        if model is None or scaler is None:
            return jsonify({"error": f"Model or scaler for stock {stock_name} not found"}), 404

        # Preprocess the data for the stock
        scaled_data, error = preprocess_data(stock_name)
        if scaled_data is None:
            return jsonify({"error": error}), 500

        # Use the last 60 days of data for prediction
        look_back = 60
        last_60_days = scaled_data[-look_back:]  # Get the last 60 days
        last_60_days = np.expand_dims(last_60_days, axis=0)  # Reshape for model input

        # Predict the next day's closing price using the model
        predicted_price_scaled = model.predict(last_60_days)

        # Inverse the scaling to get the real predicted price
        predicted_price = scaler.inverse_transform(np.concatenate((predicted_price_scaled, np.zeros((1, scaled_data.shape[1] - 1))), axis=1))[:, 0][0]

        # Return the predicted price in the response
        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=4960)
