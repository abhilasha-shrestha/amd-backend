import os
import glob
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from pathlib import Path

def cleanup_old_models():
    """Remove previous .pkl and .mat model files to avoid conflicts"""
    model_files = glob.glob('*.pkl') + glob.glob('*.mat')
    for file in model_files:
        try:
            os.remove(file)
            print(f"Removed old file: {file}")
        except FileNotFoundError:
            print(f"File not found (already removed?): {file}")
        except PermissionError:
            print(f"Permission denied while deleting {file}. Close any programs using it.")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

def train_and_save_models():
    """Main function to train and save all models"""
    # 1. Load and prepare data
    df = pd.read_csv("sensor_data.csv")
    
    # 2. Create categorical column
    water_quality = pd.cut(
        df["pH"],
        bins=[0, 3.0, 4.5, 7.0],
        labels=["Dangerous", "Warning", "Safe"]
    )
    df["water_quality"] = water_quality.astype('category')
    
    # 3. Create features and labels
    df["pH_rolling_avg"] = df["pH"].rolling(window=3).mean()
    X_pm = df[["pH", "turbidity", "orp", "pH_rolling_avg"]].dropna()
    y_pm = df["pH"].loc[X_pm.index]
    y_class = df["water_quality"].cat.codes.loc[X_pm.index]
    
    # 4. Train models
    models = {
        "model_pm1.pkl": RandomForestRegressor().fit(X_pm, y_pm),
        "model_class1.pkl": RandomForestClassifier().fit(X_pm, y_class),
        "model_anomaly1.pkl": IsolationForest(contamination=0.05).fit(X_pm)
    }
    
    # 5. Save models
    for filename, model in models.items():
        joblib.dump(model, filename)
        print(f"Created new model: {filename}")
    


if __name__ == "__main__":
    cleanup_old_models()
    train_and_save_models()
    print("Model training completed successfully!")
