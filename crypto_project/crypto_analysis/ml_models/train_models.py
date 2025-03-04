import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import torch
from crypto_analysis.ml_models.utils import (
    save_best_params,
    load_best_params,
    load_and_preprocess_data_by_currency,
    select_features,
    save_model_with_metadata,
    compute_relative_metrics,
    get_stacking_input,
)
from crypto_analysis.ml_models.gru_transformer import (
    train_gru_attention,
    train_transformer,
)
from crypto_analysis.ml_models.xgboost_lightgbm import train_xgboost_and_lightgbm
from crypto_analysis.ml_models.prophet_arima import train_prophet, train_arima
from crypto_analysis.ml_models.stacking import train_stacking

BASE_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


def train_model(X, y, model_type, params):
    if not params:
        raise ValueError("Параметры модели должны быть переданы извне.")

    split_index = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    if model_type.lower() == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params.update({"tree_method": "gpu_hist", "device": "cuda"})
        model = xgb.train(
            params, dtrain, num_boost_round=params.get("n_estimators", 100)
        )

        model.set_param({"device": "cuda", "predictor": "gpu_predictor"})
        y_pred = model.predict(dval)

    elif model_type.lower() == "lgb":
        params.update({"device": "gpu"})
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, valid_sets=[val_data], verbose_eval=False)
        y_pred = model.predict(X_val)
    else:
        raise ValueError("model_type должен быть 'xgb' или 'lgb'")

    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_val, y_pred)
    print(f"{model_type.upper()} validation MSE: {mse:.4f}")
    return model


def train_xgboost_and_lightgbm(df, params, framework, target):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
        df = df[df["date"] >= five_years_ago]
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
    cols_to_drop = ["cryptocurrency", "date", "news_hour", "date_y", "entities"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    if "close_price" not in df.columns:
        raise KeyError(
            f"Колонка 'close_price' отсутствует в DataFrame. Доступные колонки: {df.columns.tolist()}"
        )
    df["lag_1h"] = df["close_price"].shift(4)
    df["lag_24h"] = df["close_price"].shift(96)
    df = df.dropna()

    df["price_1h"] = df["lag_1h"]
    df["price_24h"] = df["lag_24h"]
    X = df.drop(columns=["close_price", "price_1h", "price_24h"])

    if target:
        y = df[target]
        horizon = "1 час" if target == "price_1h" else "24 часа"
        print(f"Обучение {framework.upper()} для предсказания цены через {horizon}:")
        return train_model(X, y, model_type=framework, params=params)

    y_1h = df["price_1h"]
    y_24h = df["price_24h"]
    models = {}
    for timeframe, y in [("1h", y_1h), ("24h", y_24h)]:
        print(f"Обучение {framework.upper()} для предсказания цены через {timeframe}:")
        models[f"{framework}_{timeframe}"] = train_model(
            X, y, model_type=framework, params=params
        )
    return models


def train_and_save_models():
    load_dotenv()
    print("Начинаю обновление моделей для каждой криптовалюты...")
    data_by_currency = load_and_preprocess_data_by_currency()
    if not data_by_currency:
        raise ValueError("Нет данных для обработки по криптовалютам.")

    cryptopairs = os.getenv("CRYPTOPAIRS")
    if cryptopairs:
        pairs_list = [pair.strip().replace("/", "_") for pair in cryptopairs.split(",")]
        data_by_currency = {
            cur: data for cur, data in data_by_currency.items() if cur in pairs_list
        }
        print(f"✅ Будут обучаться модели только для пар: {pairs_list}")
    else:
        print(
            "✅ Переменная CRYPTOPAIRS не установлена – будут обучаться модели для всех криптовалют."
        )

    tscv = TimeSeriesSplit(n_splits=10)
    param_grid_xgb = {
        "n_estimators": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750],
        "max_depth": [5, 6, 7, 8, 9, 10, 11, 12],
        "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "gamma": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "min_child_weight": [1, 2, 3, 4, 5, 6],
        "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        "reg_lambda": [0.5, 1, 1.25, 1.5, 1.75, 2],
    }
    param_grid_lgb = {
        "n_estimators": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750],
        "max_depth": [10, 15, 17, 20, 23, 25, 30],
        "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        "num_leaves": [100, 150, 170, 190, 210, 230, 250, 300],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "bagging_fraction": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "bagging_freq": [0, 5, 10, 15],
        "feature_fraction": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "min_child_samples": [10, 20, 25, 30, 35, 40, 50],
        "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        "reg_lambda": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
    }

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", device="cuda"
    )
    lgb_model = lgb.LGBMRegressor(
        objective="regression", device="gpu", force_col_wise=True, verbose=-1
    )

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
                n_jobs=1,
                n_iter=50,
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

        best_params_xgb = {}
        best_params_lgb = {}
        for horizon, target_col in [
            ("1h", "close_price_1h"),
            ("24h", "close_price_24h"),
        ]:
            params_file_xgb = os.path.join(
                BASE_MODELS_DIR, f"best_params_xgb_{horizon}_{currency}.json"
            )
            best_params_xgb[horizon] = get_best_params(
                xgb_model,
                param_grid_xgb,
                X_train_features,
                df_train_std[target_col],
                params_file_xgb,
                f"XGBoost ({horizon})",
            )
            params_file_lgb = os.path.join(
                BASE_MODELS_DIR, f"best_params_lgb_{horizon}_{currency}.json"
            )
            best_params_lgb[horizon] = get_best_params(
                lgb_model,
                param_grid_lgb,
                X_train_features,
                df_train_std[target_col],
                params_file_lgb,
                f"LightGBM ({horizon})",
            )

        gru_models, transformer_models = {}, {}
        for horizon, (X_train, y_train) in {
            "1h": (X_train_hourly, y_train_hourly),
            "24h": (X_train_4h, y_train_4h),
        }.items():
            split_index = int(0.8 * len(X_train))
            X_tr, X_val = X_train.iloc[:split_index], X_train.iloc[split_index:]
            y_tr, y_val = y_train.iloc[:split_index], y_train.iloc[split_index:]
            print(f"Обучение GRU для горизонта {horizon} для {currency}:")
            gru_models[horizon] = train_gru_attention(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                early_stopping_rounds=20,
                seq_len=25,
                epochs=200,
                batch_size=32,
                lr=0.005,
                dropout=0.1,
                impute_method="ffill",
                hidden_size=96,
                num_layers=3,
                device=device,
            )
            print(f"Обучение Transformer для горизонта {horizon} для {currency}:")
            transformer_models[horizon] = train_transformer(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                early_stopping_rounds=20,
                seq_len=15,
                epochs=200,
                batch_size=64,
                lr=0.005,
                dropout=0.2,
                impute_method="ffill",
                d_model=64,
                num_layers=2,
                nhead=4,
                device=device,
            )

        xgb_models, lgb_models = {}, {}
        for horizon, target_col in [
            ("1h", "close_price_1h"),
            ("24h", "close_price_24h"),
        ]:
            print(f"Обучение XGBoost для горизонта {horizon} для {currency}:")
            best_params_xgb[horizon]["tree_method"] = "hist"
            best_params_xgb[horizon]["device"] = "cuda"
            xgb_models[horizon] = train_xgboost_and_lightgbm(
                df_train_std,
                params=best_params_xgb[horizon],
                framework="xgb",
                target=target_col,
            )
            print(f"Обучение LightGBM для горизонта {horizon} для {currency}:")
            best_params_lgb[horizon]["device"] = "gpu"
            lgb_models[horizon] = train_xgboost_and_lightgbm(
                df_train_std,
                params=best_params_lgb[horizon],
                framework="lgb",
                target=target_col,
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
            epochs=200,
            lr=0.005,
            device=device,
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
        for model_name, horizon, model, target in [
            ("gru", "1h", gru_models["1h"], "close_price_1h"),
            ("gru", "24h", gru_models["24h"], "close_price_24h"),
            ("transformer", "1h", transformer_models["1h"], "close_price_1h"),
            ("transformer", "24h", transformer_models["24h"], "close_price_24h"),
            ("xgboost", "1h", xgb_models["1h"], "close_price_1h"),
            ("xgboost", "24h", xgb_models["24h"], "close_price_24h"),
            ("lightgbm", "1h", lgb_models["1h"], "close_price_1h"),
            ("lightgbm", "24h", lgb_models["24h"], "close_price_24h"),
        ]:
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
            (
                transformer_models["1h"],
                "transformer",
                "1h",
                currency,
                metrics.get("transformer_1h"),
            ),
            (
                transformer_models["24h"],
                "transformer",
                "24h",
                currency,
                metrics.get("transformer_24h"),
            ),
            (xgb_models["1h"], "xgboost", "1h", currency, metrics.get("xgboost_1h")),
            (xgb_models["24h"], "xgboost", "24h", currency, metrics.get("xgboost_24h")),
            (lgb_models["1h"], "lightgbm", "1h", currency, metrics.get("lightgbm_1h")),
            (
                lgb_models["24h"],
                "lightgbm",
                "24h",
                currency,
                metrics.get("lightgbm_24h"),
            ),
            (model_prophet, "prophet", currency, currency, None),
            (model_arima, "arima", currency, currency, None),
        ]
        for mdl, mdl_type, horizon, cur, met in models_to_save:
            save_model_with_metadata(mdl, mdl_type, horizon, cur, common_meta, met)

        print(f"✅ Модели для {currency} обучены и сохранены.")
    print("\n✅ Все модели обучены и сохранены для всех криптовалют.")
