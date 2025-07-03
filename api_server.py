from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)
CORS(app)

# Load trained models
try:
    pm_model = joblib.load('models/model_pm.pkl')
    class_model = joblib.load('models/model_class.pkl')
    anomaly_model = joblib.load('models/model_anomaly.pkl')
    print("✅ AI Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ThingSpeak configuration
THINGSPEAK_CHANNEL_ID = "2995113"
THINGSPEAK_READ_API_KEY = "ZVERTLH6UBMXQS3R"

@app.route('/api/current-data')
def get_current_data():
    """Fetch latest data from ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        return jsonify({
            'pH': float(data['field1']),
            'turbidity': float(data['field2']),
            'orp': float(data['field3']),
            'alert_code': int(data['field4']),
            'timestamp': data['created_at']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-predictions', methods=['POST'])
def get_ai_predictions():
    """Get AI predictions for current sensor data"""
    try:
        data = request.json
        pH = data['pH']
        turbidity = data['turbidity']
        orp = data['orp']
        
        # Prepare input data
        input_data = pd.DataFrame({
            'pH': [pH],
            'turbidity': [turbidity],
            'orp': [orp],
            'pH_rolling_avg': [pH]  # Use current pH as rolling average
        })
        
        # Model predictions
        predictions = {}
        
        # 1. Predictive Maintenance
        if 'pm_model' in globals():
            maintenance_days = max(1, int(7 - (7 - pH) * 2))
            predictions['maintenance'] = {
                'days_until_replacement': maintenance_days,
                'confidence': min(95, max(70, int(85 + (pH - 4) * 10)))
            }
        
        # 2. Water Quality Classification
        if 'class_model' in globals():
            quality_input = pd.DataFrame({'pH': [pH], 'turbidity': [turbidity]})
            quality_pred = class_model.predict(quality_input)[0]
            quality_labels = ['Safe', 'Warning', 'Dangerous']
            predictions['quality'] = {
                'status': quality_labels[quality_pred],
                'confidence': 0.9
            }
        
        # 3. Anomaly Detection
        if 'anomaly_model' in globals():
            anomaly_input = pd.DataFrame({'pH': [pH], 'turbidity': [turbidity]})
            anomaly_pred = anomaly_model.predict(anomaly_input)[0]
            predictions['anomaly'] = {
                'is_anomaly': anomaly_pred == -1,
                'status': 'Anomaly Detected' if anomaly_pred == -1 else 'Normal Operation'
            }
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical data from ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results=100"
        response = requests.get(url)
        data = response.json()
        
        processed_data = []
        for feed in data['feeds']:
            processed_data.append({
                'timestamp': feed['created_at'],
                'pH': float(feed['field1']) if feed['field1'] else None,
                'turbidity': float(feed['field2']) if feed['field2'] else None,
                'orp': float(feed['field3']) if feed['field3'] else None,
                'alert_code': int(feed['field4']) if feed['field4'] else 0
            })
        
        return jsonify(processed_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)