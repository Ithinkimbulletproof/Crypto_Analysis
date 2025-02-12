import os
import joblib
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

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
            df = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
            df.sort_values("date", inplace=True)
            if file == "unified_data.csv":
                if "close_price" in df.columns and "close_price_24h" not in df.columns:
                    df["close_price_1h"] = df["close_price"].shift(-4)
                    df["close_price_24h"] = df["close_price"].shift(-96)
            data[file] = df
            print(f"✅ Загружен {file}")
        else:
            raise FileNotFoundError(f"❌ Файл {file_path} не найден.")
    return data


def select_features(X, y, var_threshold=1e-4, corr_threshold=0.05):
    variances = X.var()
    features_high_variance = variances[variances > var_threshold].index.tolist()
    X = X[features_high_variance]

    corr_1h = X.corrwith(y["close_price_1h"]).abs()
    corr_24h = X.corrwith(y["close_price_24h"]).abs()

    selected_features = set(corr_1h[corr_1h > corr_threshold].index.tolist()) | set(
        corr_24h[corr_24h > corr_threshold].index.tolist()
    )
    selected_features = list(selected_features)
    print("✅ Отобраны признаки:", selected_features)
    return X[selected_features]


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
        f"✅ Модель {name}_{tag} сохранена по пути: {model_path} с метаданными: {meta_path}"
    )


def train_and_save_models():
    print("Начинаю обновление моделей...")
    data = load_and_preprocess_data()
    df_train_std = data["processed_train_std.csv"]

    X_train = df_train_std.select_dtypes(include=["number"]).copy()
    y_train = df_train_std[["close_price_1h", "close_price_24h"]]

    X_train = select_features(X_train, y_train)

    print("Начинаю гиперпараметрическую оптимизацию для XGBoost...")
    param_grid_xgb = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    grid_search_xgb = GridSearchCV(
        xgb_model, param_grid_xgb, cv=tscv, scoring="neg_mean_squared_error"
    )
    grid_search_xgb.fit(X_train, y_train["close_price_1h"])
    best_params_xgb = grid_search_xgb.best_params_
    print("✅ Лучшие параметры для XGBoost:", best_params_xgb)

    print("Начинаю гиперпараметрическую оптимизацию для LightGBM...")
    param_grid_lgb = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [31, 50, 70],
        "subsample": [0.7, 1.0],
    }
    lgb_model = lgb.LGBMRegressor(objective="regression")
    grid_search_lgb = GridSearchCV(
        lgb_model, param_grid_lgb, cv=tscv, scoring="neg_mean_squared_error"
    )
    grid_search_lgb.fit(X_train, y_train["close_price_1h"])
    best_params_lgb = grid_search_lgb.best_params_
    print("✅ Лучшие параметры для LightGBM:", best_params_lgb)

    print("Обучение базовых моделей...")
    base_models = {
        "lstm": train_lstm(X_train, df_train_std["close_price"]),
        "transformer": train_transformer(X_train, df_train_std["close_price"]),
        "xgboost": train_xgboost_and_lightgbm(
            X_train, params=best_params_xgb, framework="xgboost"
        ),
        "lightgbm": train_xgboost_and_lightgbm(
            X_train, params=best_params_lgb, framework="lightgbm"
        ),
        "prophet": train_prophet(
            df_train_std.set_index("date")["close_price"], "unified"
        ),
        "arima": train_arima(df_train_std.set_index("date")["close_price"], "unified"),
    }
    for name, model in base_models.items():
        save_model_with_metadata(
            model, name, "unified", {"unified_data.csv": data["unified_data.csv"]}
        )

    print("Обучение стэкинговой модели с 2 выходами...")
    stacking_model = train_stacking(base_models, X_train, y_train, epochs=100, lr=0.01)
    save_model_with_metadata(
        stacking_model,
        "stacking",
        "unified",
        {"unified_data.csv": data["unified_data.csv"]},
    )
    print("✅ Все модели обучены и сохранены.")


def main():
    print("Начинаю обновление моделей...")
    train_and_save_models()
    print("Обновление завершено.")
