import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ✅ Step 1: Load dataset
folder_path = r"C:\Users\Sahil khan(Machine E\Music\New folder (3)\CMAPSSData"
train_file = os.path.join(folder_path, "train_FD001.txt")
test_file = os.path.join(folder_path, "test_FD001.txt")
rul_file = os.path.join(folder_path, "RUL_FD001.txt")

# Columns based on NASA C-MAPSS description
col_names = [
    "unit", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
    "sensor_19", "sensor_20", "sensor_21"
]

# Load training and test data
train_df = pd.read_csv(train_file, sep=" ", header=None, names=col_names)
train_df.dropna(axis=1, how='all', inplace=True)

test_df = pd.read_csv(test_file, sep=" ", header=None, names=col_names)
test_df.dropna(axis=1, how='all', inplace=True)

rul_df = pd.read_csv(rul_file, sep=" ", header=None, names=["RUL"])

print("✅ Data Loaded Successfully!")
print(f"Training Shape: {train_df.shape}")
print(f"Testing Shape: {test_df.shape}")
print(train_df.head())

# ✅ Step 2: Compute Remaining Useful Life (RUL)
rul_train = train_df.groupby("unit")["time_in_cycles"].max().reset_index()
rul_train.columns = ["unit", "max_cycle"]
train_df = train_df.merge(rul_train, on="unit", how="left")
train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]

# ✅ Step 3: Feature Selection
features = [col for col in train_df.columns if col not in ["unit", "time_in_cycles", "max_cycle", "RUL"]]
X = train_df[features]
y = train_df["RUL"]

# ✅ Step 4: Split and Scale Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ✅ Step 5: Train Model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Step 6: Predictions and Evaluation
y_pred = model.predict(X_val_scaled)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# ✅ Step 7: Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted Remaining Useful Life (RUL)")
plt.grid(True)
plt.show()

# ✅ Step 8: Save model (optional)
import joblib
model_path = os.path.join(folder_path, "turbofan_RUL_model.pkl")
joblib.dump(model, model_path)
print(f"\n✅ Model saved successfully at: {model_path}")
