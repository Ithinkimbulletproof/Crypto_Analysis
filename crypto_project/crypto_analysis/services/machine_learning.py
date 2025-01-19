import os
import joblib
import logging
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from crypto_analysis.models import TechAnalysed
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = "models"


def save_predictions(predictions, file_name="predictions.csv"):
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", file_name)
    predictions.to_csv(file_path, index=False)
    logger.info(f"Предсказания сохранены в {file_path}")


def load_processed_data(cryptocurrency=None, period=None):
    query = TechAnalysed.objects.all()
    if cryptocurrency:
        query = query.filter(cryptocurrency=cryptocurrency)
    if period:
        query = query.filter(period=period)

    data = list(query.values('date', 'cryptocurrency', 'period', 'close_price', 'high_price', 'low_price',
                             'price_change_24h', 'SMA_30', 'volatility_30', 'SMA_90', 'volatility_90',
                             'SMA_180', 'volatility_180', 'predicted_signal', 'target'))

    if not data:
        logger.warning(f"Данные для {cryptocurrency} в периоде {period} не найдены.")
        return None

    df = pd.DataFrame(data)

    return df


def prepare_data(df, target_col="close", features_to_exclude=None):
    if features_to_exclude is None:
        features_to_exclude = ["date", "cryptocurrency", "period"]

    features = df.drop(columns=features_to_exclude + [target_col], errors="ignore")
    target = df[target_col]

    features = features.fillna(0)
    target = target.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


def train_model(X_train, y_train, model_type="XGBoost"):
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(random_state=42)
    elif model_type == "LightGBM":
        model = lgb.LGBMRegressor(random_state=42)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        logger.error(f"Неизвестный тип модели: {model_type}")
        return None

    param_grid = {
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 10],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 10],
        },
    }

    if model_type in param_grid:
        grid_search = GridSearchCV(
            model, param_grid=param_grid[model_type], cv=3, verbose=2, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    return model


def train_lstm(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    train_data = torch.tensor(X_train, dtype=torch.float32)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    train_labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    test_labels = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]
    hidden_layer_size = 50
    output_size = 1

    model = LSTMModel(input_size, hidden_layer_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]

            optimizer.zero_grad()

            output = model(batch_data.unsqueeze(1))
            loss = criterion(output, batch_labels)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model


def train_arima(df, target_col="close"):
    model = ARIMA(df[target_col], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


def evaluate_model(model, X_test, y_test, model_type="XGBoost"):
    if isinstance(model, LSTMModel):
        model.eval()
        with torch.no_grad():
            predictions = model(
                torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
            )
            predictions = predictions.squeeze()

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info(f"Оценка модели LSTM: MSE={mse:.4f} R2={r2:.4f}")
        return mse, r2

    predictions = model.predict(X_test)
    if model_type == "LogisticRegression":
        predictions = predictions.round()
    accuracy = (
        accuracy_score(y_test, predictions)
        if model_type == "LogisticRegression"
        else None
    )
    mse = (
        mean_squared_error(y_test, predictions)
        if model_type != "LogisticRegression"
        else None
    )
    r2 = r2_score(y_test, predictions) if model_type != "LogisticRegression" else None

    logger.info(
        f"Оценка модели {model_type}: MSE={mse:.4f} R2={r2:.4f} Accuracy={accuracy:.4f}"
    )
    return mse, r2, accuracy


def save_model(model, model_name="crypto_forecast_model.pkl"):
    os.makedirs(MODEL_PATH, exist_ok=True)
    file_path = os.path.join(MODEL_PATH, model_name)
    if isinstance(model, LSTMModel):
        torch.save(model.state_dict(), file_path)
    else:
        joblib.dump(model, file_path)
    logger.info(f"Модель сохранена в {file_path}")


def generate_predictions_short_term(model, X_test, y_test):
    predictions = model.predict(X_test)
    data = {
        "cryptocurrency": X_test.index.get_level_values("cryptocurrency"),
        "date": X_test.index.get_level_values("date"),
        "predicted_change": predictions - y_test,
        "predicted_close": predictions,
    }
    return pd.DataFrame(data)


def generate_predictions_long_term(model, df, target_col="close"):
    predictions = model.predict(df[target_col])
    dates = pd.date_range(start=df.index[-1], periods=len(predictions), freq="D")

    data = {
        "cryptocurrency": df["cryptocurrency"].iloc[-1],
        "date": dates,
        "predicted_close": predictions,
        "predicted_change": predictions - df[target_col].iloc[-1],
    }
    return pd.DataFrame(data)


def process_machine_learning():
    file_path = "processed_with_technical_analysis.csv"
    df = load_processed_data(file_path)
    if df is None:
        return

    X_train, X_test, y_train, y_test = prepare_data(df)

    logger.info("Обучение моделей краткосрочного прогноза (24 часа)")
    short_term_predictions = []
    models = ["XGBoost", "LightGBM", "LogisticRegression"]
    for model_type in models:
        model = train_model(X_train, y_train, model_type)
        mse, r2, accuracy = evaluate_model(model, X_test, y_test, model_type)
        save_model(model, model_name=f"{model_type}_short_term_model.pkl")

        predictions = generate_predictions_short_term(model, X_test, y_test)
        predictions["model"] = model_type
        short_term_predictions.append(predictions)

    short_term_df = pd.concat(short_term_predictions)
    save_predictions(short_term_df, file_name="short_term_predictions.csv")

    logger.info("Обучение моделей долгосрочного прогноза")
    lstm_model = train_lstm(X_train.values, y_train, X_test.values, y_test)
    save_model(lstm_model, model_name="LSTM_long_term_model.pth")

    arima_model = train_arima(df)
    save_model(arima_model, model_name="ARIMA_long_term_model.pkl")

    long_term_predictions = generate_predictions_long_term(arima_model, df)
    save_predictions(long_term_predictions, file_name="long_term_predictions.csv")


if __name__ == "__main__":
    process_machine_learning()
