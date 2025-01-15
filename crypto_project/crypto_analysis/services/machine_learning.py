import os
import joblib
import logging
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = "models"


def load_processed_data(file_path="processed_with_technical_analysis.csv"):
    if not os.path.exists(file_path):
        logger.warning(f"Файл {file_path} не найден.")
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        logger.warning(f"Файл {file_path} пуст.")
        return None
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


def train_lstm(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
    )
    return model


def train_arima(df, target_col="close"):
    model = ARIMA(df[target_col], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


def evaluate_model(model, X_test, y_test, model_type="XGBoost"):
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
    joblib.dump(model, file_path)
    logger.info(f"Модель сохранена в {file_path}")


def process_machine_learning():
    file_path = "processed_with_technical_analysis.csv"
    df = load_processed_data(file_path)
    if df is None:
        return

    X_train, X_test, y_train, y_test = prepare_data(df)

    logger.info("Обучение моделей краткосрочного прогноза (24 часа)")
    models = ["XGBoost", "LightGBM", "LogisticRegression"]
    for model_type in models:
        model = train_model(X_train, y_train, model_type)
        mse, r2, accuracy = evaluate_model(model, X_test, y_test, model_type)
        save_model(model, model_name=f"{model_type}_short_term_model.pkl")

    logger.info("Обучение моделей долгосрочного прогноза")
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = train_lstm(X_train_lstm, y_train, X_test_lstm, y_test)
    save_model(lstm_model, model_name="LSTM_long_term_model.pkl")

    arima_model = train_arima(df)
    save_model(arima_model, model_name="ARIMA_long_term_model.pkl")


if __name__ == "__main__":
    process_machine_learning()
