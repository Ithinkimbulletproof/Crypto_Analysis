import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
from crypto_analysis.ml_models.gru_transformer import (
    train_gru_attention,
    train_transformer,
)
from crypto_analysis.ml_models.xgboost_lightgbm import train_xgboost_and_lightgbm
from crypto_analysis.ml_models.prophet_arima import train_prophet, train_arima
from crypto_analysis.ml_models.stacking import train_stacking, get_model_predictions

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


def compute_relative_metrics(model, X, y_true, model_name=None, raw_data=None):
    if model_name is not None and (
        model_name.startswith("xgboost") or model_name.startswith("lightgbm")
    ):
        y_pred = get_model_predictions(model, model_name, X, raw_data=raw_data)
        if model_name.endswith("_1h"):
            y_pred = y_pred[:, 0]
        elif model_name.endswith("_24h"):
            y_pred = y_pred[:, 1]
        else:
            y_pred = y_pred[:, 0]
    else:
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
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(1)
            with torch.no_grad():
                y_pred_tensor = model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy().squeeze()

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    elif isinstance(y_true, pd.Series):
        y_true = y_true.values
    else:
        y_true = np.array(y_true)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    min_len = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if y_pred.shape[1] == y_true.shape[1]:
        metrics = {}
        target_names = (
            ["1h", "24h"]
            if y_true.shape[1] == 2
            else [str(i) for i in range(y_true.shape[1])]
        )
        for i in range(y_true.shape[1]):
            mask = y_true[:, i] != 0
            mape = (
                np.mean(np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i]))
                * 100
            )
            metrics[f"MAPE_{target_names[i]}"] = mape
        return metrics
    elif y_pred.shape[1] == 1:
        mask = y_true[:, 0] != 0
        mape = (
            np.mean(np.abs((y_true[mask, 0] - y_pred[mask, 0]) / y_true[mask, 0])) * 100
        )
        return {"MAPE": mape}
    else:
        raise ValueError(
            f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}"
        )


def get_stacking_input(base_models, X, raw_data=None):
    preds = []
    for name, model in base_models.items():
        if name.startswith("xgboost") or name.startswith("lightgbm"):
            pred = get_model_predictions(model, name, X, raw_data=raw_data)
        else:
            pred = get_model_predictions(model, name, X)
        preds.append(pred)
    min_rows = min(pred.shape[0] for pred in preds)
    preds_aligned = [pred[:min_rows] for pred in preds]
    X_stack = np.hstack(preds_aligned)
    return X_stack


def train_and_save_models():
    load_dotenv()

    print("Начинаю обновление моделей для каждой криптовалюты...")
    data_by_currency = load_and_preprocess_data_by_currency()
    if not data_by_currency:
        raise ValueError("Нет данных для обработки по криптовалютам.")

    cryptopairs = os.getenv("CRYPTOPAIRS")
    if cryptopairs:
        pairs_list = [pair.strip().replace("/", "_") for pair in cryptopairs.split(",")]
        data_by_currency = {cur: data for cur, data in data_by_currency.items() if cur in pairs_list}
        print(f"✅ Будут обучаться модели только для пар: {pairs_list}")
    else:
        print("✅ Переменная CRYPTOPAIRS не установлена – будут обучаться модели для всех криптовалют.")

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid_xgb = {
        "n_estimators": [300, 350, 400, 450, 500, 550, 600, 650, 700],
        "max_depth": [7, 8, 9, 10, 11],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "gamma": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "min_child_weight": [1, 2, 3, 4, 5],
        "reg_alpha": [0, 0.005, 0.01, 0.05, 0.1],
        "reg_lambda": [1, 1.25, 1.5, 1.75, 2],
    }

    param_grid_lgb = {
        "n_estimators": [300, 350, 400, 450, 500, 550, 600, 650, 700],
        "max_depth": [15, 17, 20, 23, 25],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "num_leaves": [150, 170, 190, 210, 230, 250],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "bagging_freq": [0, 5, 10],
        "feature_fraction": [0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "min_child_samples": [20, 25, 30, 35, 40, 50],
        "reg_alpha": [0, 0.005, 0.01, 0.05, 0.1],
        "reg_lambda": [0, 0.005, 0.01, 0.05, 0.1],
    }

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    lgb_model = lgb.LGBMRegressor(objective="regression", force_col_wise=True, verbose=-1)

    def get_best_params(model, param_grid, X, y, params_file, model_desc):
        if os.path.exists(params_file):
            best_params = load_best_params(params_file)
            print(f"✅ Загружены оптимальные параметры для {model_desc} для {currency}")
        else:
            random_search = RandomizedSearchCV(
                model,
                param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=3,
                n_iter=150,
                random_state=42,
            )
            random_search.fit(X, y)
            best_params = random_search.best_params_
            save_best_params(best_params, params_file)
            print(
                f"✅ Оптимизация {model_desc} для {currency} завершена и параметры сохранены"
            )
        return best_params

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

        params_file_xgb_1h = os.path.join(
            MODEL_DIR, f"best_params_xgb_1h_{currency}.json"
        )
        best_params_xgb_1h = get_best_params(
            xgb_model,
            param_grid_xgb,
            X_train_features,
            df_train_std["close_price_1h"],
            params_file_xgb_1h,
            "XGBoost (1h)",
        )
        params_file_xgb_24h = os.path.join(
            MODEL_DIR, f"best_params_xgb_24h_{currency}.json"
        )
        best_params_xgb_24h = get_best_params(
            xgb_model,
            param_grid_xgb,
            X_train_features,
            df_train_std["close_price_24h"],
            params_file_xgb_24h,
            "XGBoost (24h)",
        )

        params_file_lgb_1h = os.path.join(
            MODEL_DIR, f"best_params_lgb_1h_{currency}.json"
        )
        best_params_lgb_1h = get_best_params(
            lgb_model,
            param_grid_lgb,
            X_train_features,
            df_train_std["close_price_1h"],
            params_file_lgb_1h,
            "LightGBM (1h)",
        )
        params_file_lgb_24h = os.path.join(
            MODEL_DIR, f"best_params_lgb_24h_{currency}.json"
        )
        best_params_lgb_24h = get_best_params(
            lgb_model,
            param_grid_lgb,
            X_train_features,
            df_train_std["close_price_24h"],
            params_file_lgb_24h,
            "LightGBM (24h)",
        )

        gru_models = {}
        transformer_models = {}
        for horizon, (X_train, y_train) in {
            "1h": (X_train_hourly, y_train_hourly),
            "24h": (X_train_4h, y_train_4h),
        }.items():
            print(f"Обучение GRU для горизонта {horizon} для {currency}:")
            gru_models[horizon] = train_gru_attention(
                X_train,
                y_train,
                seq_len=25,
                epochs=50,
                batch_size=32,
                lr=0.005,
                dropout=0.1,
                impute_method="ffill",
                hidden_size=96,
                num_layers=3,
            )
            print(f"Обучение Transformer для горизонта {horizon} для {currency}:")
            transformer_models[horizon] = train_transformer(
                X_train,
                y_train,
                seq_len=15,
                epochs=33,
                batch_size=64,
                lr=0.001,
                dropout=0.2,
                impute_method="ffill",
                d_model=64,
                num_layers=2,
                nhead=4,
            )

        xgb_models, lgb_models = {}, {}
        for horizon in ["1h", "24h"]:
            target = "close_price_1h" if horizon == "1h" else "close_price_24h"
            print(f"Обучение XGBoost для горизонта {horizon} для {currency}:")
            best_params = best_params_xgb_1h if horizon == "1h" else best_params_xgb_24h
            xgb_models[horizon] = train_xgboost_and_lightgbm(
                df_train_std, params=best_params, framework="xgb", target=target
            )
            print(f"Обучение LightGBM для горизонта {horizon} для {currency}:")
            best_params = best_params_lgb_1h if horizon == "1h" else best_params_lgb_24h
            lgb_models[horizon] = train_xgboost_and_lightgbm(
                df_train_std, params=best_params, framework="lgb", target=target
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
            "gru_1h": gru_models["1h"],
            "gru_24h": gru_models["24h"],
            "transformer_1h": transformer_models["1h"],
            "transformer_24h": transformer_models["24h"],
            "xgboost_1h": xgb_models["1h"],
            "xgboost_24h": xgb_models["24h"],
            "lightgbm_1h": lgb_models["1h"],
            "lightgbm_24h": lgb_models["24h"],
            "prophet": model_prophet,
            "arima": model_arima,
        }

        print(f"Обучение стэкинговой модели для {currency} с 2 выходами...")
        stacking_model = train_stacking(
            base_models,
            X_train_features,
            df_train_std[["close_price_1h", "close_price_24h"]],
            raw_data=raw_data,
            epochs=100,
            lr=0.01,
        )

        X_stack_for_metrics = get_stacking_input(
            base_models, X_train_features, raw_data=raw_data
        )
        metrics_stack = compute_relative_metrics(
            stacking_model,
            X_stack_for_metrics,
            df_train_std[["close_price_1h", "close_price_24h"]],
        )
        metrics = {}
        model_metric_pairs = [
            ("gru", "1h", gru_models["1h"], "close_price_1h"),
            ("gru", "24h", gru_models["24h"], "close_price_24h"),
            ("transformer", "1h", transformer_models["1h"], "close_price_1h"),
            ("transformer", "24h", transformer_models["24h"], "close_price_24h"),
            ("xgboost", "1h", xgb_models["1h"], "close_price_1h"),
            ("xgboost", "24h", xgb_models["24h"], "close_price_24h"),
            ("lightgbm", "1h", lgb_models["1h"], "close_price_1h"),
            ("lightgbm", "24h", lgb_models["24h"], "close_price_24h"),
        ]
        for model_name, horizon, model, target in model_metric_pairs:
            if model_name in ["xgboost", "lightgbm"]:
                metrics[f"{model_name}_{horizon}"] = compute_relative_metrics(
                    model,
                    X_train_features,
                    df_train_std[target],
                    model_name=f"{model_name}_{horizon}",
                    raw_data=raw_data,
                )
            else:
                metrics[f"{model_name}_{horizon}"] = compute_relative_metrics(
                    model, X_train_features, df_train_std[target]
                )

        print(f"\nРезультаты для {currency}:")
        print(f"Stacking: {metrics_stack}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

            common_meta = {"unified": data["unified"]}
            models_to_save = [
                (stacking_model, "stacking", "unified", currency, metrics_stack),
                (gru_models["1h"], "gru", "1h", currency, metrics.get("gru_1h")),
                (gru_models["24h"], "gru", "24h", currency, metrics.get("gru_24h")),
                (transformer_models["1h"], "transformer", "1h", currency, metrics.get("transformer_1h")),
                (transformer_models["24h"], "transformer", "24h", currency, metrics.get("transformer_24h")),
                (xgb_models["1h"], "xgboost", "1h", currency, metrics.get("xgboost_1h")),
                (xgb_models["24h"], "xgboost", "24h", currency, metrics.get("xgboost_24h")),
                (lgb_models["1h"], "lightgbm", "1h", currency, metrics.get("lightgbm_1h")),
                (lgb_models["24h"], "lightgbm", "24h", currency, metrics.get("lightgbm_24h")),
                (model_prophet, "prophet", currency, currency, None),
                (model_arima, "arima", currency, currency, None),
            ]

            for mdl, mdl_type, horizon, cur, met in models_to_save:
                save_model_with_metadata(mdl, mdl_type, horizon, cur, common_meta, met)

        print(f"✅ Модели для {currency} обучены и сохранены.")
    print("\n✅ Все модели обучены и сохранены для всех криптовалют.")
