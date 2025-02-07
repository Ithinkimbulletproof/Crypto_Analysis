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

DATA_DIR = "data_exports"


def load_and_preprocess_data():
    files = ["processed_data_minmax.csv", "processed_data_std.csv", "unified_data.csv"]
    data = {}

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            if file == "unified_data.csv":
                df = pd.read_csv(
                    file_path, parse_dates=["date_x", "date_y"], low_memory=False
                )
                df.rename(columns={"date_x": "date"}, inplace=True)
                data[file] = df
            else:
                data[file] = pd.read_csv(
                    file_path, parse_dates=["date"], low_memory=False
                )
            print(f"✅ Загружен {file}")
        else:
            raise FileNotFoundError(f"❌ Файл {file_path} не найден.")
    return data


def save_model_with_metadata(model, name, data):
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    df_unified = data["unified_data.csv"]
    last_date = df_unified["date"].max()
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

    X_train_minmax = data["processed_data_minmax.csv"].drop(
        columns=["date", "cryptocurrency"]
    )
    y_train_minmax = X_train_minmax.pop("close_price_24h")

    X_train_std = data["processed_data_std.csv"].drop(
        columns=["date", "cryptocurrency"]
    )
    y_train_std = X_train_std.pop("close_price_24h")

    print("Обучение моделей...")

    models = {
        "lstm": train_lstm(X_train_std, y_train_std),
        "transformer": train_transformer(X_train_std, y_train_std),
        "xgboost": train_xgboost(X_train_std, y_train_std),
        "lightgbm": train_lightgbm(X_train_std, y_train_std),
        "prophet": train_prophet(
            data["unified_data.csv"].set_index("date")["close_price"]
        ),
        "arima": train_arima(data["unified_data.csv"].set_index("date")["close_price"]),
    }

    for name, model in models.items():
        save_model_with_metadata(model, name, data)

    stacking_model = train_stacking(models, X_train_std, y_train_std)
    save_model_with_metadata(stacking_model, "stacking", data)

    print("Все модели дообучены и сохранены.")


def main():
    print("Начинаю обновление моделей...")
    train_and_save_models()
    print("Обновление завершено.")


if __name__ == "__main__":
    main()
