# Predict Remaining Operational Cycles of Turbofan Engine (Predictive Maintenance Domain)

## 🔍 Project Overview
This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using NASA’s **C-MAPSS dataset**.  
By leveraging **Machine Learning (Random Forest Regression)** and **time-series analysis**, it provides early insights for maintenance planning — reducing failure risks and improving operational efficiency.

## 🧠 Technologies Used
- Python 🐍  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn (Random Forest Regressor)  
- Joblib (for model saving)

## 📂 Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation Data  
Available from the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

**Files used:**
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## ⚙️ How It Works
1. Load and preprocess the dataset.  
2. Compute RUL (Remaining Useful Life) for each engine.  
3. Train the Random Forest regression model.  
4. Evaluate predictions using MAE, RMSE, and R² metrics.  
5. Visualize **Actual vs Predicted RUL**.

## 📊 Model Performance
| Metric | Value (approx.) |
|---------|----------------|
| MAE     | ~XX.XX |
| RMSE    | ~XX.XX |
| R²      | ~0.XX |

## 🚀 Usage
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python turbofan_RUL_prediction.py
