import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from crypto_analysis.ml_models.stacking import get_model_predictions

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
