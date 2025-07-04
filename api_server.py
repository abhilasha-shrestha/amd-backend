from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
import json
import logging

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

@app.route('/api/current-data')
def get_current_data():
    """Fetch latest data from ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_API_KEY}"
        logger.info(f"Fetching data from: {url}")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Handle potential None values from ThingSpeak
        ph_value = float(data.get('field1', 0)) if data.get('field1') else 6.5
        turbidity_value = float(data.get('field2', 0)) if data.get('field2') else 500
        orp_value = float(data.get('field3', 0)) if data.get('field3') else -100
        alert_code = int(data.get('field4', 0)) if data.get('field4') else 0
        
        result = {
            'pH': ph_value,
            'turbidity': turbidity_value,
            'orp': orp_value,
            'alert_code': alert_code,
            'timestamp': data.get('created_at', ''),
            'status': 'success'
        }
        
        logger.info(f"Successfully fetched data: {result}")
        return jsonify(result)
        
    except requests.RequestException as e:
        logger.error(f"ThingSpeak API error: {e}")
        return jsonify({
            'error': 'Failed to fetch data from ThingSpeak',
            'details': str(e),
            'fallback_data': {
                'pH': 6.5,
                'turbidity': 500,
                'orp': -100,
                'alert_code': 0
            }
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-predictions', methods=['POST','GET'])
def get_ai_predictions():
    """Get AI predictions for current sensor data"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        pH = float(data.get('pH', 6.5))
        turbidity = float(data.get('turbidity', 500))
        orp = float(data.get('orp', -100))
        
        logger.info(f"Received data for prediction: pH={pH}, turbidity={turbidity}, orp={orp}")
        
        predictions = {}
        
        # 1. Predictive Maintenance
        try:
            if models_loaded['pm_model']:
                pm_input = pd.DataFrame({
                    'pH': [pH],
                    'turbidity': [turbidity],
                    'orp': [orp],
                    'pH_rolling_avg': [pH]
                })
                
                pm_pred = pm_model.predict(pm_input)[0]
                maintenance_days = max(1, int(pm_pred))
                confidence = min(95, max(70, int(85 + (pH - 4) * 5)))
                
                predictions['maintenance'] = {
                    'days_until_replacement': maintenance_days,
                    'confidence': confidence
                }
            else:
                maintenance_days = max(1, int(7 - (7 - pH) * 2))
                predictions['maintenance'] = {
                    'days_until_replacement': maintenance_days,
                    'confidence': 75
                }
        except Exception as e:
            logger.error(f"PM model error: {e}")
            predictions['maintenance'] = {
                'days_until_replacement': 5,
                'confidence': 50,
                'error': 'Model unavailable'
            }
        
        # 2. Water Quality Classification
        try:
            if models_loaded['class_model']:
                class_input = pd.DataFrame({
                    'pH': [pH], 
                    'turbidity': [turbidity]
                })
                
                quality_pred = class_model.predict(class_input)[0]
                quality_labels = ['Safe', 'Warning', 'Dangerous']
                
                predictions['quality'] = {
                    'status': quality_labels[min(len(quality_labels)-1, max(0, quality_pred))],
                    'confidence': 0.9
                }
            else:
                if pH < 4.0 or turbidity > 1500:
                    status = 'Dangerous'
                elif pH < 5.0 or turbidity > 1000:
                    status = 'Warning'
                else:
                    status = 'Safe'
                
                predictions['quality'] = {
                    'status': status,
                    'confidence': 0.7
                }
        except Exception as e:
            logger.error(f"Classification model error: {e}")
            predictions['quality'] = {
                'status': 'Unknown',
                'confidence': 0.5,
                'error': 'Model unavailable'
            }
        
        # 3. Anomaly Detection
        try:
            if models_loaded['anomaly_model']:
                anomaly_input = pd.DataFrame({
                    'pH': [pH], 
                    'turbidity': [turbidity]
                })
                
                anomaly_pred = anomaly_model.predict(anomaly_input)[0]
                is_anomaly = bool(anomaly_pred == -1)  # ✅ FIX: Convert np.bool_ to bool
                
                predictions['anomaly'] = {
                    'is_anomaly': is_anomaly,
                    'status': 'Anomaly Detected' if is_anomaly else 'Normal Operation'
                }
            else:
                is_anomaly = pH < 3.0 or pH > 9.0 or turbidity > 2000
                predictions['anomaly'] = {
                    'is_anomaly': is_anomaly,
                    'status': 'Anomaly Detected' if is_anomaly else 'Normal Operation'
                }
        except Exception as e:
            logger.error(f"Anomaly model error: {e}")
            predictions['anomaly'] = {
                'is_anomaly': False,
                'status': 'Detection unavailable',
                'error': 'Model unavailable'
            }
        
        logger.info(f"Generated predictions: {predictions}")
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Failed to generate predictions',
            'details': str(e),
            'fallback_predictions': {
                'maintenance': {'days_until_replacement': 5, 'confidence': 50},
                'quality': {'status': 'Unknown', 'confidence': 0.5},
                'anomaly': {'is_anomaly': False, 'status': 'Detection unavailable'}
            }
        }), 500

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical data from ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        processed_data = []
        for feed in data.get('feeds', []):
            processed_data.append({
                'timestamp': feed.get('created_at', ''),
                'pH': float(feed.get('field1', 0)) if feed.get('field1') else None,
                'turbidity': float(feed.get('field2', 0)) if feed.get('field2') else None,
                'orp': float(feed.get('field3', 0)) if feed.get('field3') else None,
                'alert_code': int(feed.get('field4', 0)) if feed.get('field4') else 0
            })
        
        return jsonify(processed_data)
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models-status')
def get_models_status():
    """Get status of loaded AI models"""
    return jsonify({
        'models_loaded': models_loaded,
        'total_models': len(models_loaded),
        'loaded_count': sum(models_loaded.values())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)