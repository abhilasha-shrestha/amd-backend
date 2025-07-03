import pandas as pd

# 1. Load your sensor data
df = pd.read_csv("sensor_data.csv")  # Update filename if different

# 2. Clean data (example)
df = df.dropna()  # Remove missing values

# 3. Feature engineering - rolling averages
df["pH_rolling_avg"] = df["pH"].rolling(window=3).mean()
df["turbidity_rolling_avg"] = df["turbidity"].rolling(window=3).mean()

# 4. Create labels for AI
df["maintenance_needed"] = (df["pH"] > 6.5).astype(int)  # 1 if limestone needs replacement
df["water_quality"] = pd.cut(
    df["pH"],
    bins=[0, 3.0, 4.5, 7.0],
    labels=["Dangerous", "Warning", "Safe"]
)

# 5. Save processed data
df.to_csv("labeled_data.csv", index=False)
print("Data preprocessing complete! Saved as labeled_data.csv")