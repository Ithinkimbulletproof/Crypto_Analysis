import os
import joblib
import json
import pandas as pd
import numpy as np
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


def save_best_params(params, filename):
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)


def load_best_params(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_and_preprocess_data_by_currency():
    data_by_currency = {}
    for filename in os.listdir(DATA_DIR):
        if filename.startswith("unified_data_") and filename.endswith(".csv"):
            currency = filename[len("unified_data_") : -len(".csv")]
            unified_file = os.path.join(DATA_DIR, f"unified_data_{currency}.csv")
            processed_train_std_file = os.path.join(
                DATA_DIR, f"processed_train_std_{currency}.csv"
            )
            processed_train_minmax_file = os.path.join(
                DATA_DIR, f"processed_train_minmax_{currency}.csv"
            )
            if (
                os.path.exists(unified_file)
                and os.path.exists(processed_train_std_file)
                and os.path.exists(processed_train_minmax_file)
            ):
                df_unified = pd.read_csv(
                    unified_file, parse_dates=["date"], low_memory=False
                )
                df_train_std = pd.read_csv(
                    processed_train_std_file, parse_dates=["date"], low_memory=False
                )
                df_train_minmax = pd.read_csv(
                    processed_train_minmax_file, parse_dates=["date"], low_memory=False
                )
                df_unified.sort_values("date", inplace=True)
                df_train_std.sort_values("date", inplace=True)
                df_train_minmax.sort_values("date", inplace=True)
                data_by_currency[currency] = {
                    "unified": df_unified,
                    "processed_train_std": df_train_std,
                    "processed_train_minmax": df_train_minmax,
                }
    return data_by_currency


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


def save_model_with_metadata(model, name, tag, currency, data, metrics=None):
    model_path = os.path.join(MODEL_DIR, f"{name}_{tag}_{currency}.pkl")
    joblib.dump(model, model_path)
    df_unified = data["unified"]
    last_date = df_unified["date"].max()
    if pd.notnull(last_date):
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_date = "unknown"
    metadata = {
        "model_name": f"{name}_{tag}_{currency}",
        "forecast_tag": tag,
        "currency": currency,
        "last_processed_date": last_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metrics is not None:
        metadata["relative_metrics"] = metrics
    meta_path = os.path.join(MODEL_DIR, f"{name}_{tag}_{currency}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(
        f"✅ Модель {name}_{tag}_{currency} сохранена по пути: {model_path} с метаданными: {meta_path}"
    )


def compute_relative_metrics(model, X, y_true):
    try:
        y_pred = model.predict(X)
    except AttributeError:
        import torch

        model.eval()
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy().squeeze()

    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    else:
        y_true = np.array(y_true)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAPE": mape}


def train_and_save_models():
    print("Начинаю обновление моделей для каждой криптовалюты...")
    data_by_currency = load_and_preprocess_data_by_currency()
    if not data_by_currency:
        raise ValueError("Нет данных для обработки по криптовалютам.")

    for currency, data in data_by_currency.items():
        print(f"\nОбучение моделей для {currency}...")
        df_train_std = data["processed_train_std"].copy()

        if "cryptocurrency" in df_train_std.columns:
            df_train_std.drop(columns=["cryptocurrency"], inplace=True)

        df_train_std.columns = df_train_std.columns.str.replace(" ", "_")
        df_train_std["date"] = pd.to_datetime(df_train_std["date"])
        raw_data = df_train_std.copy()

        df_hourly = df_train_std[df_train_std["date"].dt.minute == 0].copy()
        df_hourly["close_price_1h"] = df_hourly["close_price"].shift(-1)
        df_4h = df_train_std[
            (df_train_std["date"].dt.minute == 0)
            & (df_train_std["date"].dt.hour % 4 == 0)
        ].copy()
        df_4h["close_price_24h"] = df_4h["close_price"].shift(-6)

        X_train_hourly = df_hourly.select_dtypes(include=["number"]).copy()
        y_train_hourly = df_hourly["close_price_1h"]
        X_train_4h = df_4h.select_dtypes(include=["number"]).copy()
        y_train_4h = df_4h["close_price_24h"]

        X_train_features = select_features(
            df_train_std.select_dtypes(include=["number"]).copy(),
            df_train_std[["close_price_1h", "close_price_24h"]],
        )
        common_features = X_train_features.columns.intersection(X_train_hourly.columns)
        X_train_hourly = X_train_hourly[common_features]
        X_train_4h = X_train_4h[common_features]

        param_grid_xgb = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.7, 1.0],
        }
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

        params_file_xgb_1h = os.path.join(
            MODEL_DIR, f"best_params_xgb_1h_{currency}.json"
        )
        if os.path.exists(params_file_xgb_1h):
            best_params_xgb_1h = load_best_params(params_file_xgb_1h)
            print(f"✅ Загружены оптимальные параметры для XGBoost (1h) для {currency}")
        else:
            grid_search_xgb = GridSearchCV(
                xgb_model, param_grid_xgb, cv=tscv, scoring="neg_mean_squared_error"
            )
            grid_search_xgb.fit(X_train_features, df_train_std["close_price_1h"])
            best_params_xgb_1h = grid_search_xgb.best_params_
            save_best_params(best_params_xgb_1h, params_file_xgb_1h)
            print(
                f"✅ Оптимизация XGBoost (1h) для {currency} завершена и параметры сохранены"
            )

        params_file_xgb_24h = os.path.join(
            MODEL_DIR, f"best_params_xgb_24h_{currency}.json"
        )
        if os.path.exists(params_file_xgb_24h):
            best_params_xgb_24h = load_best_params(params_file_xgb_24h)
            print(
                f"✅ Загружены оптимальные параметры для XGBoost (24h) для {currency}"
            )
        else:
            grid_search_xgb_24h = GridSearchCV(
                xgb_model, param_grid_xgb, cv=tscv, scoring="neg_mean_squared_error"
            )
            grid_search_xgb_24h.fit(X_train_features, df_train_std["close_price_24h"])
            best_params_xgb_24h = grid_search_xgb_24h.best_params_
            save_best_params(best_params_xgb_24h, params_file_xgb_24h)
            print(
                f"✅ Оптимизация XGBoost (24h) для {currency} завершена и параметры сохранены"
            )

        param_grid_lgb = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [50, 100, 150],
            "subsample": [0.7, 1.0],
        }
        lgb_model = lgb.LGBMRegressor(
            objective="regression", force_col_wise=True, verbose=-1
        )

        params_file_lgb_1h = os.path.join(
            MODEL_DIR, f"best_params_lgb_1h_{currency}.json"
        )
        if os.path.exists(params_file_lgb_1h):
            best_params_lgb_1h = load_best_params(params_file_lgb_1h)
            print(
                f"✅ Загружены оптимальные параметры для LightGBM (1h) для {currency}"
            )
        else:
            grid_search_lgb = GridSearchCV(
                lgb_model, param_grid_lgb, cv=tscv, scoring="neg_mean_squared_error"
            )
            grid_search_lgb.fit(X_train_features, df_train_std["close_price_1h"])
            best_params_lgb_1h = grid_search_lgb.best_params_
            save_best_params(best_params_lgb_1h, params_file_lgb_1h)
            print(
                f"✅ Оптимизация LightGBM (1h) для {currency} завершена и параметры сохранены"
            )

        params_file_lgb_24h = os.path.join(
            MODEL_DIR, f"best_params_lgb_24h_{currency}.json"
        )
        if os.path.exists(params_file_lgb_24h):
            best_params_lgb_24h = load_best_params(params_file_lgb_24h)
            print(
                f"✅ Загружены оптимальные параметры для LightGBM (24h) для {currency}"
            )
        else:
            grid_search_lgb_24h = GridSearchCV(
                lgb_model, param_grid_lgb, cv=tscv, scoring="neg_mean_squared_error"
            )
            grid_search_lgb_24h.fit(X_train_features, df_train_std["close_price_24h"])
            best_params_lgb_24h = grid_search_lgb_24h.best_params_
            save_best_params(best_params_lgb_24h, params_file_lgb_24h)
            print(
                f"✅ Оптимизация LightGBM (24h) для {currency} завершена и параметры сохранены"
            )

        print(f"Обучение LSTM для горизонта 1h для {currency}:")
        lstm_model_1h = train_lstm(
            X_train_hourly,
            y_train_hourly,
            seq_len=10,
            epochs=10,
            batch_size=64,
            lr=0.001,
            dropout=0.2,
        )
        print(f"Обучение LSTM для горизонта 24h для {currency}:")
        lstm_model_24h = train_lstm(
            X_train_4h,
            y_train_4h,
            seq_len=10,
            epochs=10,
            batch_size=64,
            lr=0.001,
            dropout=0.2,
        )
        lstm_models = {"1h": lstm_model_1h, "24h": lstm_model_24h}

        print(f"Обучение Transformer для горизонта 1h для {currency}:")
        transformer_model_1h = train_transformer(
            X_train_hourly,
            y_train_hourly,
            seq_len=10,
            epochs=10,
            batch_size=64,
            lr=0.001,
            dropout=0.2,
        )
        print(f"Обучение Transformer для горизонта 24h для {currency}:")
        transformer_model_24h = train_transformer(
            X_train_4h,
            y_train_4h,
            seq_len=10,
            epochs=10,
            batch_size=64,
            lr=0.001,
            dropout=0.2,
        )
        transformer_models = {"1h": transformer_model_1h, "24h": transformer_model_24h}

        print(f"Обучение XGBoost для горизонта 1h для {currency}:")
        model_xgb_1h = train_xgboost_and_lightgbm(
            df_train_std,
            params=best_params_xgb_1h,
            framework="xgboost",
            target="close_price_1h",
        )
        print(f"Обучение XGBoost для горизонта 24h для {currency}:")
        model_xgb_24h = train_xgboost_and_lightgbm(
            df_train_std,
            params=best_params_xgb_24h,
            framework="xgboost",
            target="close_price_24h",
        )

        print(f"Обучение LightGBM для горизонта 1h для {currency}:")
        model_lgb_1h = train_xgboost_and_lightgbm(
            df_train_std,
            params=best_params_lgb_1h,
            framework="lightgbm",
            target="close_price_1h",
        )
        print(f"Обучение LightGBM для горизонта 24h для {currency}:")
        model_lgb_24h = train_xgboost_and_lightgbm(
            df_train_std,
            params=best_params_lgb_24h,
            framework="lightgbm",
            target="close_price_24h",
        )

        print(f"Обучение Prophet для {currency}:")
        model_prophet = train_prophet(
            df_train_std.set_index("date")["close_price"], forecast_tag=currency
        )
        print(f"Обучение ARIMA для {currency}:")
        model_arima = train_arima(
            df_train_std.set_index("date")["close_price"], forecast_tag=currency
        )

        base_models = {
            "lstm_1h": lstm_models["1h"],
            "lstm_24h": lstm_models["24h"],
            "transformer_1h": transformer_models["1h"],
            "transformer_24h": transformer_models["24h"],
            "xgboost_1h": model_xgb_1h,
            "xgboost_24h": model_xgb_24h,
            "lightgbm_1h": model_lgb_1h,
            "lightgbm_24h": model_lgb_24h,
            "prophet": model_prophet,
            "arima": model_arima,
        }

        print(f"Обучение стэкинговой модели для {currency} с 2 выходами...")
        stacking_model = train_stacking(
            base_models,
            X_train_features,
            df_train_std[["close_price_1h", "close_price_24h"]],
            raw_data=raw_data,
            epochs=10,
            lr=0.01,
        )

        metrics_stack = compute_relative_metrics(
            stacking_model, X_train_features, df_train_std["close_price_1h"]
        )
        metrics_lstm1h = compute_relative_metrics(
            lstm_models["1h"], X_train_features, df_train_std["close_price_1h"]
        )
        metrics_lstm24h = compute_relative_metrics(
            lstm_models["24h"], X_train_features, df_train_std["close_price_24h"]
        )
        metrics_trans1h = compute_relative_metrics(
            transformer_models["1h"], X_train_features, df_train_std["close_price_1h"]
        )
        metrics_trans24h = compute_relative_metrics(
            transformer_models["24h"], X_train_features, df_train_std["close_price_24h"]
        )
        metrics_xgb1h = compute_relative_metrics(
            model_xgb_1h, X_train_features, df_train_std["close_price_1h"]
        )
        metrics_xgb24h = compute_relative_metrics(
            model_xgb_24h, X_train_features, df_train_std["close_price_24h"]
        )
        metrics_lgb1h = compute_relative_metrics(
            model_lgb_1h, X_train_features, df_train_std["close_price_1h"]
        )
        metrics_lgb24h = compute_relative_metrics(
            model_lgb_24h, X_train_features, df_train_std["close_price_24h"]
        )

        save_model_with_metadata(
            stacking_model,
            "stacking",
            "unified",
            currency,
            {"unified": data["unified"]},
            metrics_stack,
        )
        save_model_with_metadata(
            lstm_models["1h"],
            "lstm",
            "1h",
            currency,
            {"unified": data["unified"]},
            metrics_lstm1h,
        )
        save_model_with_metadata(
            lstm_models["24h"],
            "lstm",
            "24h",
            currency,
            {"unified": data["unified"]},
            metrics_lstm24h,
        )
        save_model_with_metadata(
            transformer_models["1h"],
            "transformer",
            "1h",
            currency,
            {"unified": data["unified"]},
            metrics_trans1h,
        )
        save_model_with_metadata(
            transformer_models["24h"],
            "transformer",
            "24h",
            currency,
            {"unified": data["unified"]},
            metrics_trans24h,
        )
        save_model_with_metadata(
            model_xgb_1h,
            "xgboost",
            "1h",
            currency,
            {"unified": data["unified"]},
            metrics_xgb1h,
        )
        save_model_with_metadata(
            model_xgb_24h,
            "xgboost",
            "24h",
            currency,
            {"unified": data["unified"]},
            metrics_xgb24h,
        )
        save_model_with_metadata(
            model_lgb_1h,
            "lightgbm",
            "1h",
            currency,
            {"unified": data["unified"]},
            metrics_lgb1h,
        )
        save_model_with_metadata(
            model_lgb_24h,
            "lightgbm",
            "24h",
            currency,
            {"unified": data["unified"]},
            metrics_lgb24h,
        )
        save_model_with_metadata(
            model_prophet, "prophet", currency, currency, {"unified": data["unified"]}
        )
        save_model_with_metadata(
            model_arima, "arima", currency, currency, {"unified": data["unified"]}
        )
        print(f"✅ Модели для {currency} обучены и сохранены.")
    print("\n✅ Все модели обучены и сохранены для всех криптовалют.")
