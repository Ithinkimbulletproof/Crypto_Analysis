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
    files = [
        "processed_train_minmax.csv",
        "processed_train_std.csv",
        "unified_data.csv",
    ]
    data = {}
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            if file == "unified_data.csv":
                df = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
                df.sort_values("date", inplace=True)
                if "close_price" in df.columns and "close_price_24h" not in df.columns:
                    df["close_price_1h"] = df["close_price"].shift(-4)
                    df["close_price_24h"] = df["close_price"].shift(-96)
                data[file] = df
            else:
                data[file] = pd.read_csv(
                    file_path, parse_dates=["date"], low_memory=False
                )
            print(f"✅ Загружен {file}")
        else:
            raise FileNotFoundError(f"❌ Файл {file_path} не найден.")
    return data


def save_model_with_metadata(model, name, tag, data):
    model_path = os.path.join(MODEL_DIR, f"{name}_{tag}.pkl")
    joblib.dump(model, model_path)
    df_unified = data["unified_data.csv"]
    last_date = df_unified["date"].max()
    if pd.notnull(last_date):
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_date = "unknown"
    metadata = {
        "model_name": f"{name}_{tag}",
        "forecast_tag": tag,
        "last_processed_date": last_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(MODEL_DIR, f"{name}_{tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(
        f"Модель {name}_{tag} сохранена по пути: {model_path} с метаданными: {meta_path}"
    )


def train_and_save_models():
    data = load_and_preprocess_data()
    df_minmax = data["processed_train_minmax.csv"]
    df_std = data["processed_train_std.csv"]
    df_unified = data["unified_data.csv"]

    X_train = df_std.select_dtypes(include=["number"]).copy()
    y_train = pd.DataFrame(
        {
            "close_price_1h": df_unified["close_price_1h"],
            "close_price_24h": df_unified["close_price_24h"],
        }
    )

    print("Обучение базовых моделей...")
    base_models = {
        "lstm": train_lstm(X_train, df_unified["close_price"]),
        "transformer": train_transformer(X_train, df_unified["close_price"]),
        "xgboost": train_xgboost_and_lightgbm(X_train),
        "lightgbm": train_xgboost_and_lightgbm(X_train),
        "prophet": train_prophet(
            df_unified.set_index("date")["close_price"], "unified"
        ),
        "arima": train_arima(df_unified.set_index("date")["close_price"], "unified"),
    }
    for name, model in base_models.items():
        save_model_with_metadata(
            model, name, "unified", {"unified_data.csv": df_unified}
        )

    print("Обучение стэкинговой модели с 2 выходами...")
    stacking_model = train_stacking(base_models, X_train, y_train, epochs=100, lr=0.01)
    save_model_with_metadata(
        stacking_model, "stacking", "unified", {"unified_data.csv": df_unified}
    )
    print("Все модели обучены и сохранены.")


def main():
    print("Начинаю обновление моделей...")
    train_and_save_models()
    print("Обновление завершено.")


if __name__ == "__main__":
    main()
