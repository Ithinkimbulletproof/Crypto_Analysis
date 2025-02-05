import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class StackingModel(nn.Module):
    def __init__(self, input_size):
        super(StackingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_stacking(models, X_train, y_train, epochs=100, lr=0.01):
    preds = []
    for name, model in models.items():
        if name in ["prophet", "arima"]:
            if name == "prophet":
                future = model.make_future_dataframe(periods=1, freq="D")
                forecast = model.predict(future)
                pred = forecast["yhat"].iloc[-1]
            else:
                pred = model.predict(n_periods=1)[0]
            pred_array = np.full((len(X_train), 1), pred)
        else:
            try:
                X_np = X_train.values
            except AttributeError:
                X_np = X_train
            pred_array = model.predict(X_np)
            if pred_array.ndim == 1:
                pred_array = pred_array.reshape(-1, 1)
        preds.append(pred_array)
    X_stack = np.hstack(preds)
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

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
            print(f"Stacking Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return stacking_model


def predict_and_save(models, stacking_model, X, current_date, db_model):
    preds = []
    for name, model in models.items():
        if name in ["prophet", "arima"]:
            if name == "prophet":
                future = model.make_future_dataframe(periods=1, freq="D")
                forecast = model.predict(future)
                pred = forecast["yhat"].iloc[-1]
            else:
                pred = model.predict(n_periods=1)[0]
            pred_array = np.full((len(X), 1), pred)
        else:
            try:
                X_np = X.values
            except AttributeError:
                X_np = X
            pred_array = model.predict(X_np)
            if pred_array.ndim == 1:
                pred_array = pred_array.reshape(-1, 1)
        preds.append(pred_array)
    X_stack = np.hstack(preds)
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32)
    stacking_model.eval()
    with torch.no_grad():
        final_pred = stacking_model(X_stack_tensor).numpy().flatten()
    try:
        price_now = X["close_price"].iloc[-1]
    except KeyError:
        price_now = None
    price_24h = final_pred[-1]
    change_percent = ((price_24h - price_now) / price_now * 100) if price_now else None
    prediction_confidence = 0.9

    prediction = db_model.objects.create(
        current_price=price_now,
        price_24h=price_24h,
        change_percent=change_percent,
        prediction_date=current_date,
        confidence=prediction_confidence,
    )
    prediction.save()
    print("Предсказание сохранено в БД.")
    return final_pred, prediction
