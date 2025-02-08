import os
import joblib
import json
import pandas as pd
from datetime import datetime
from crypto_analysis.ml_models.lstm_transformer import train_lstm, train_transformer
from crypto_analysis.ml_models.xgboost_lightgbm import train_xgboost_and_lightgbm
from crypto_analysis.ml_models.prophet_arima import train_prophet, train_arima
from crypto_analysis.ml_models.stacking import train_stacking

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

DATA_DIR = "data_exports"

def load_and_preprocess_data():
    files = ["processed_data_minmax.csv", "processed_data_std.csv", "unified_data.csv"]
    data = {}
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            if file == "unified_data.csv":
                df = pd.read_csv(file_path, parse_dates=["date_x", "date_y"], low_memory=False)
                df.rename(columns={"date_x": "date"}, inplace=True)
                data[file] = df
            else:
                data[file] = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
            print(f"✅ Загружен {file}")
        else:
            raise FileNotFoundError(f"❌ Файл {file_path} не найден.")
    return data

def save_model_with_metadata(model, name, horizon, data):
    model_path = os.path.join(MODEL_DIR, f"{name}_{horizon}.pkl")
    joblib.dump(model, model_path)

    df_unified = data["unified_data.csv"]
    last_date = df_unified["date"].max()
    if pd.notnull(last_date):
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_date = "unknown"

    metadata = {
        "model_name": f"{name}_{horizon}",
        "forecast_horizon": horizon,
        "last_processed_date": last_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(MODEL_DIR, f"{name}_{horizon}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Модель {name}_{horizon} сохранена по пути: {model_path} с метаданными: {meta_path}")

def filter_by_interval(df, horizon):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if horizon == "1h":
        return df[df["date"].dt.minute == 0]
    elif horizon == "24h":
        return df[(df["date"].dt.minute == 0) & (df["date"].dt.hour % 4 == 0)]
    else:
        return df

def train_and_save_models():
    data = load_and_preprocess_data()

    df_minmax = data["processed_data_minmax.csv"]
    df_std = data["processed_data_std.csv"]
    df_unified = data["unified_data.csv"]

    df_minmax_1h = filter_by_interval(df_minmax, "1h")
    df_std_1h = filter_by_interval(df_std, "1h")
    df_unified_1h = filter_by_interval(df_unified, "1h")

    df_minmax_24h = filter_by_interval(df_minmax, "24h")
    df_std_24h = filter_by_interval(df_std, "24h")
    df_unified_24h = filter_by_interval(df_unified, "24h")

    drop_cols_1h = ["date", "cryptocurrency", "date_y", "entities", "close_price_24h", "news_hour"]
    X_train_minmax_1h = df_minmax_1h.drop(columns=drop_cols_1h)
    y_train_minmax_1h = X_train_minmax_1h.pop("close_price")
    X_train_std_1h = df_std_1h.drop(columns=drop_cols_1h)
    y_train_std_1h = X_train_std_1h.pop("close_price")

    drop_cols_24h = ["date", "cryptocurrency", "date_y", "entities", "close_price", "news_hour"]
    X_train_minmax_24h = df_minmax_24h.drop(columns=drop_cols_24h)
    y_train_minmax_24h = X_train_minmax_24h.pop("close_price_24h")
    X_train_std_24h = df_std_24h.drop(columns=drop_cols_24h)
    y_train_std_24h = X_train_std_24h.pop("close_price_24h")

    print("Обучение моделей...")

    for (
        horizon,
        X_train_minmax,
        X_train_std,
        y_train_minmax,
        y_train_std,
        df_unified_filtered,
    ) in [
        (
            "1h",
            X_train_minmax_1h,
            X_train_std_1h,
            y_train_minmax_1h,
            y_train_std_1h,
            df_unified_1h,
        ),
        (
            "24h",
            X_train_minmax_24h,
            X_train_std_24h,
            y_train_minmax_24h,
            y_train_std_24h,
            df_unified_24h,
        ),
    ]:
        models = {
            "lstm": train_lstm(X_train_std, y_train_std),
            "transformer": train_transformer(X_train_std, y_train_std),
            "xgboost": train_xgboost_and_lightgbm(X_train_std),
            "lightgbm": train_xgboost_and_lightgbm(X_train_std),
            "prophet": train_prophet(df_unified_filtered.set_index("date")["close_price"], horizon),
            "arima": train_arima(df_unified_filtered.set_index("date")["close_price"], horizon),
        }
        for name, model in models.items():
            save_model_with_metadata(model, name, horizon, {"unified_data.csv": df_unified_filtered})
        stacking_model = train_stacking(models, X_train_std, y_train_std)
        save_model_with_metadata(stacking_model, "stacking", horizon, {"unified_data.csv": df_unified_filtered})

    print("Все модели дообучены и сохранены.")

def main():
    print("Начинаю обновление моделей...")
    train_and_save_models()
    print("Обновление завершено.")

if __name__ == "__main__":
    main()
