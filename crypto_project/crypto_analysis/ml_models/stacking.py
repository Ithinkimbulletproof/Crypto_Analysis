import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from crypto_analysis.models import CryptoPrediction
from crypto_analysis.ml_models.xgboost_lightgbm import prepare_features


class StackingModel(nn.Module):
    def __init__(self, input_size):
        super(StackingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc_out = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


def get_model_predictions(model, name, X, raw_data=None):
    if name.startswith("xgboost") or name.startswith("lightgbm"):
        if raw_data is not None:
            X_prepared = prepare_features(raw_data)
        else:
            X_prepared = prepare_features(X)
        cols_to_remove = ["close_price", "price_1h", "price_24h"]
        X_prepared = X_prepared.drop(columns=cols_to_remove, errors="ignore")
        if name.startswith("xgboost"):
            booster = model.get_booster()
            expected_features = booster.feature_names
            X_prepared = X_prepared.reindex(columns=expected_features, fill_value=0)
        elif name.startswith("lightgbm"):
            expected_features = model.feature_name_
            X_prepared = X_prepared.reindex(columns=expected_features, fill_value=0)
        pred_array = model.predict(X_prepared)
        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(-1, 1)
        if name.endswith("_1h"):
            pred_array = np.hstack([pred_array, np.zeros_like(pred_array)])
        elif name.endswith("_24h"):
            pred_array = np.hstack([np.zeros_like(pred_array), pred_array])
        else:
            if pred_array.shape[1] == 1:
                pred_array = np.hstack([pred_array, pred_array])
        return pred_array

    elif isinstance(model, torch.nn.Module):
        X_np = X.values if hasattr(X, "values") else X
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        model.eval()
        with torch.no_grad():
            pred_tensor = model(X_tensor)
        pred_array = pred_tensor.numpy()
        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(-1, 1)
        if name.endswith("_1h"):
            pred_array = np.hstack([pred_array, np.zeros_like(pred_array)])
        elif name.endswith("_24h"):
            pred_array = np.hstack([np.zeros_like(pred_array), pred_array])
        else:
            if pred_array.shape[1] == 1:
                pred_array = np.hstack([pred_array, pred_array])
        return pred_array

    elif name in ["prophet", "arima"]:
        if name == "prophet":
            future = model.make_future_dataframe(periods=24, freq="h")
            forecast = model.predict(future)
            pred_1h = forecast["yhat"].iloc[-24]
            pred_24h = forecast["yhat"].iloc[-1]
        else:
            pred_1h = model.predict(n_periods=1).iloc[0]
            pred_24h = model.predict(n_periods=24).iloc[-1]
        pred_array = np.column_stack(
            (np.full(len(X), pred_1h), np.full(len(X), pred_24h))
        )
        return pred_array

    else:
        X_np = X.values if hasattr(X, "values") else X
        pred_array = model.predict(X_np)
        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(-1, 1)
        if pred_array.shape[1] == 1:
            pred_array = np.hstack([pred_array, pred_array])
        return pred_array


def train_stacking(models, X_train, y_train, raw_data=None, epochs=100, lr=0.01):
    preds = []
    for name, model in models.items():
        if name.startswith("xgboost") or name.startswith("lightgbm"):
            pred = get_model_predictions(model, name, X_train, raw_data=raw_data)
        else:
            pred = get_model_predictions(model, name, X_train)
        preds.append(pred)
    min_rows = min(pred.shape[0] for pred in preds)
    preds_aligned = [pred[:min_rows] for pred in preds]
    X_stack = np.hstack(preds_aligned)
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values[:min_rows], dtype=torch.float32)
    input_size = X_stack.shape[1]
    stacking_model = StackingModel(input_size=input_size)
    optimizer = optim.Adam(stacking_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    stacking_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = stacking_model(X_stack_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Stacking Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return stacking_model


def predict_and_save(models, stacking_model, X, current_date, symbol, raw_data=None):
    preds = []
    for name, model in models.items():
        if name.startswith("xgboost") or name.startswith("lightgbm"):
            pred = get_model_predictions(model, name, X, raw_data=raw_data)
        else:
            pred = get_model_predictions(model, name, X)
        preds.append(pred)
    min_rows = min(pred.shape[0] for pred in preds)
    preds_aligned = [pred[:min_rows] for pred in preds]
    X_stack = np.hstack(preds_aligned)
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32)
    stacking_model.eval()
    with torch.no_grad():
        final_pred = stacking_model(X_stack_tensor).numpy()
    try:
        price_now = X["close_price"].iloc[-1]
    except KeyError:
        price_now = None
    price_1h = final_pred[-1, 0]
    price_24h = final_pred[-1, 1]
    change_percent_1h = (
        ((price_1h - price_now) / price_now * 100) if price_now else None
    )
    change_percent_24h = (
        ((price_24h - price_now) / price_now * 100) if price_now else None
    )
    prediction_confidence = 0.9
    prediction = CryptoPrediction.objects.create(
        symbol=symbol if isinstance(symbol, str) else ",".join(symbol),
        current_price=price_now,
        price_1h=price_1h,
        price_24h=price_24h,
        change_percent_1h=change_percent_1h,
        change_percent_24h=change_percent_24h,
        prediction_date=current_date,
        confidence=prediction_confidence,
    )
    prediction.save()
    print(f"✅ Предсказания для {symbol} на 1h и 24h сохранены в БД.")
    return final_pred, prediction
