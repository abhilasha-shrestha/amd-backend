from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
import json
import logging
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained models
models_loaded = {
    'pm_model': False,
    'class_model': False,
    'anomaly_model': False
}

try:
    pm_model = joblib.load('models/model_pm.pkl')
    models_loaded['pm_model'] = True
    logger.info("✅ Predictive Maintenance model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading PM model: {e}")

try:
    class_model = joblib.load('models/model_class.pkl')
    models_loaded['class_model'] = True
    logger.info("✅ Classification model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading Classification model: {e}")

try:
    anomaly_model = joblib.load('models/model_anomaly.pkl')
    models_loaded['anomaly_model'] = True
    logger.info("✅ Anomaly detection model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading Anomaly model: {e}")

# ThingSpeak configuration
THINGSPEAK_CHANNEL_ID = "2995113"
THINGSPEAK_READ_API_KEY = "ZVERTLH6UBMXQS3R"

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'AMD Backend API is running',
        'models_loaded': models_loaded,
        'thingspeak_channel': THINGSPEAK_CHANNEL_ID
    })

# ✅ Static test route (for testing with frontend)
@app.route('/api/current-data', methods=['GET'])
def get_static_current_data():
    """Static Example Sensor Data for Testing"""
    sensor_data = {
        "pH": 4.2,
        "turbidity": 850,
        "orp": -120,
        "alert_code": 1
    }
    return jsonify(sensor_data)

# ✅ AI Prediction Route (Retained as-is)
@app.route('/api/ai-predictions', methods=['POST'])
def get_ai_predictions():
    """Get AI predictions for current sensor data"""
    try:
        data = request.get_json()
        pH = data.get('pH')
        turbidity = data.get('turbidity')
        orp = data.get('orp')

        # Validate input
        if pH is None or turbidity is None or orp is None:
            return jsonify({'error': 'Missing sensor data'}), 400

        # Prepare input for models
        sensor_input = pd.DataFrame([[pH, turbidity, orp]], columns=['pH', 'turbidity', 'orp'])
        logger.info(f"Received sensor data for prediction: {sensor_input.to_dict()}")

        # Predictive Maintenance Prediction
        pm_prediction = pm_model.predict(sensor_input)[0] if models_loaded['pm_model'] else None

        # Classification Prediction
        class_prediction = class_model.predict(sensor_input)[0] if models_loaded['class_model'] else None

        # Anomaly Detection Prediction
        anomaly_prediction = anomaly_model.predict(sensor_input)[0] if models_loaded['anomaly_model'] else None

        logger.info(f"Predictions - PM: {pm_prediction}, Class: {class_prediction}, Anomaly: {anomaly_prediction}")

        return jsonify({
            'pm_prediction': int(pm_prediction) if pm_prediction is not None else None,
            'class_prediction': int(class_prediction) if class_prediction is not None else None,
            'anomaly_prediction': int(anomaly_prediction) if anomaly_prediction is not None else None
        })

    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# ✅ Historical Data Route (Retained as-is)
@app.route('/api/historical-data')
def get_historical_data():
    """Get historical data from ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
        params = {
            'api_key': THINGSPEAK_READ_API_KEY,
            'results': 100
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            feeds = data.get('feeds', [])

            # Extract fields and format data
            historical_data = []
            for entry in feeds:
                historical_data.append({
                    'timestamp': entry.get('created_at'),
                    'pH': entry.get('field1'),
                    'turbidity': entry.get('field2'),
                    'orp': entry.get('field3'),
                })

            logger.info(f"Fetched {len(historical_data)} records from ThingSpeak")
            return jsonify(historical_data)
        else:
            logger.error(f"ThingSpeak API Error: {response.status_code}")
            return jsonify({'error': 'Failed to fetch data from ThingSpeak'}), 500

    except Exception as e:
        logger.error(f"❌ Error fetching historical data: {e}")
        return jsonify({'error': str(e)}), 500

# ✅ Models Status Route
@app.route('/api/models-status')
def get_models_status():
    """Get status of loaded AI models"""
    return jsonify({
        'models_loaded': models_loaded,
        'total_models': len(models_loaded),
        'loaded_count': sum(models_loaded.values())
    })

# ✅ App Runner (Render/Local Friendly)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)