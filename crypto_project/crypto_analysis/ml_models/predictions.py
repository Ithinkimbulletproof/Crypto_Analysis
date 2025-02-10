import os
import joblib
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
from datetime import datetime
from crypto_analysis.ml_models.stacking import predict_and_save

load_dotenv()

stacking_model = joblib.load("models/stacking_unified.pkl")
lstm_model = joblib.load("models/lstm_unified.pkl")
transformer_model = joblib.load("models/transformer_unified.pkl")
xgboost_model = joblib.load("models/xgboost_unified.pkl")
lightgbm_model = joblib.load("models/lightgbm_unified.pkl")
prophet_model = joblib.load("models/prophet_unified.pkl")
arima_model = joblib.load("models/arima_unified.pkl")

models = {
    "lstm": lstm_model,
    "transformer": transformer_model,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
    "prophet": prophet_model,
    "arima": arima_model
}

X_actual = pd.read_csv("unified_data.csv")
X_actual["date"] = pd.to_datetime(X_actual["date"])

cutoff_48h = X_actual["date"].max() - pd.Timedelta(hours=48)
X_hourly = X_actual[X_actual["date"] >= cutoff_48h]
X_hourly = X_hourly[X_hourly["date"].dt.minute == 0]

cutoff_7d = X_actual["date"].max() - pd.Timedelta(days=7)
X_daily = X_actual[X_actual["date"] >= cutoff_7d]
X_inference = X_daily.copy()
X_inference = X_inference[X_inference["date"].dt.minute == 0]

current_date = datetime.now()
symbols = os.getenv("CRYPTOPAIRS").split(",")

final_pred, prediction = predict_and_save(models, stacking_model, X_inference, current_date, symbols)
price_1h = final_pred[-1, 0]
price_24h = final_pred[-1, 1]
print(f"Для {symbols} предсказание: на 1h = {price_1h}, на 24h = {price_24h}")
