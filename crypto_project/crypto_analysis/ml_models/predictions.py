import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from crypto_analysis.ml_models.stacking import predict_and_save


def run_predictions():
    load_dotenv()

    cryptopairs = os.getenv("CRYPTOPAIRS")
    if not cryptopairs:
        raise ValueError(
            "Переменная окружения CRYPTOPAIRS не установлена. Укажите её в файле .env"
        )

    with open("data_exports/features_list.json", "r") as f:
        features_list = json.load(f)

    stacking_model = joblib.load("models/stacking_unified.pkl")
    lstm_model_1h = joblib.load("models/lstm_1h.pkl")
    lstm_model_24h = joblib.load("models/lstm_24h.pkl")
    transformer_model_1h = joblib.load("models/transformer_1h.pkl")
    transformer_model_24h = joblib.load("models/transformer_24h.pkl")
    xgboost_model_1h = joblib.load("models/xgboost_1h.pkl")
    xgboost_model_24h = joblib.load("models/xgboost_24h.pkl")
    lightgbm_model_1h = joblib.load("models/lightgbm_1h.pkl")
    lightgbm_model_24h = joblib.load("models/lightgbm_24h.pkl")
    prophet_model = joblib.load("models/prophet_unified.pkl")
    arima_model = joblib.load("models/arima_unified.pkl")

    models = {
        "lstm_1h": lstm_model_1h,
        "lstm_24h": lstm_model_24h,
        "transformer_1h": transformer_model_1h,
        "transformer_24h": transformer_model_24h,
        "xgboost_1h": xgboost_model_1h,
        "xgboost_24h": xgboost_model_24h,
        "lightgbm_1h": lightgbm_model_1h,
        "lightgbm_24h": lightgbm_model_24h,
        "prophet": prophet_model,
        "arima": arima_model,
    }

    X_actual = pd.read_csv("data_exports/unified_data.csv", low_memory=False)
    X_actual["date"] = pd.to_datetime(X_actual["date"])
    X_actual.columns = X_actual.columns.str.replace(" ", "_")

    if "cryptocurrency" in X_actual.columns:
        X_actual = pd.get_dummies(X_actual, columns=["cryptocurrency"], drop_first=True)

    if "close_price_24h" not in X_actual.columns:
        X_actual["close_price_1h"] = X_actual["close_price"].shift(-4)
        X_actual["close_price_24h"] = X_actual["close_price"].shift(-96)
    if "volume.1" not in X_actual.columns:
        X_actual["volume.1"] = X_actual["volume"]
    if "BBANDS_Middle.1" not in X_actual.columns:
        X_actual["BBANDS_Middle.1"] = X_actual["BBANDS_Middle"]
    if "VWAP.1" not in X_actual.columns:
        X_actual["VWAP.1"] = X_actual["VWAP"]

    if "hour" not in X_actual.columns:
        X_actual["hour"] = X_actual["date"].dt.hour
    if "day_of_week" not in X_actual.columns:
        X_actual["day_of_week"] = X_actual["date"].dt.dayofweek

    cutoff_7d = X_actual["date"].max() - pd.Timedelta(days=7)
    X_daily = X_actual[X_actual["date"] >= cutoff_7d]
    X_inference = X_daily.copy()
    X_inference = X_inference[X_inference["date"].dt.minute == 0]

    try:
        X_inference_numeric = X_inference[features_list]
    except KeyError as e:
        print("Отсутствуют признаки, ожидаемые моделью:", e)
        raise

    current_date = datetime.now()
    symbols = cryptopairs.split(",")

    final_pred, prediction = predict_and_save(
        models,
        stacking_model,
        X_inference_numeric,
        current_date,
        symbols,
        raw_data=X_inference,
    )

    price_1h = final_pred[-1, 0]
    price_24h = final_pred[-1, 1]
    print(f"✅ Для {symbols} предсказание: на 1h = {price_1h}, на 24h = {price_24h}")
