import os
import joblib
import json
import pandas as pd
from datetime import datetime
from crypto_analysis.ml_models.lstm_transformer import train_lstm, train_transformer
from crypto_analysis.ml_models.xgboost_lightgbm import train_xgboost, train_lightgbm
from crypto_analysis.ml_models.prophet_arima import train_prophet, train_arima
from crypto_analysis.ml_models.stacking import train_stacking

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def load_and_preprocess_data():
    if os.path.exists("unified_data.csv"):
        df_unified = pd.read_csv("unified_data.csv", parse_dates=["date"])
        print("Загружен unified_data.csv")
    else:
        raise FileNotFoundError("Файл unified_data.csv не найден.")

    from crypto_analysis.preprocess.data_aggregator import preprocessing_data

    processed_data, _ = preprocessing_data(df_unified)
    return processed_data


def save_model_with_metadata(model, name, data):
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    last_date = data["df_original"]["date"].max()
    if pd.notnull(last_date):
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_date = "unknown"

    metadata = {
        "model_name": name,
        "last_processed_date": last_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(MODEL_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Модель {name} сохранена по пути: {model_path} с метаданными: {meta_path}")


def train_and_save_models():
    data = load_and_preprocess_data()

    X_train = data["X_train_std"]
    y_train = data["y_train_std"]

    models = {
        "lstm": train_lstm(X_train, y_train),
        "transformer": train_transformer(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
        "prophet": train_prophet(data["df_original"].set_index("date")["close_price"]),
        "arima": train_arima(data["df_original"].set_index("date")["close_price"]),
    }

    for name, model in models.items():
        save_model_with_metadata(model, name, data)

    stacking_model = train_stacking(models, X_train, y_train)
    save_model_with_metadata(stacking_model, "stacking", data)

    print("Все модели дообучены и сохранены.")


def main():
    print("Начинаю обновление моделей...")
    train_and_save_models()
    print("Обновление завершено.")


if __name__ == "__main__":
    main()
