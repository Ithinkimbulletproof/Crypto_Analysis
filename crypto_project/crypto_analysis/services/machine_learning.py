import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from django.db import transaction
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from crypto_analysis.models import (
    IndicatorData,
    ShortTermCryptoPrediction,
    LongTermCryptoPrediction,
)


def load_indicator_data(selected_indicators=None):
    query = IndicatorData.objects.all()
    if selected_indicators:
        query = query.filter(indicator_name__in=selected_indicators)

    indicators = query.values("cryptocurrency", "date", "indicator_name", "value")
    df = pd.DataFrame(indicators)
    df["date"] = pd.to_datetime(df["date"])

    df_wide = df.pivot(index=["cryptocurrency", "date"], columns="indicator_name", values="value").reset_index()

    df_wide["original_value"] = df.groupby(["cryptocurrency", "date"])["value"].mean().reset_index(drop=True)

    return df_wide


def short_term_indicators():
    return [
        "price_change_1d",
        "price_change_7d",
        "price_change_14d",
        "SMA_7",
        "SMA_14",
        "SMA_30",
        "volatility_7d",
        "volatility_14d",
        "volatility_30d",
        "RSI_7d",
        "RSI_14d",
        "RSI_30d",
        "CCI_7d",
        "CCI_14d",
        "ATR_14",
        "ATR_30",
        "Stochastic_7",
        "Stochastic_14",
        "MACD_12_26",
        "MACD_signal_9",
        "original_value",
    ]


def long_term_indicators():
    return [
        "price_change_7d",
        "price_change_14d",
        "price_change_30d",
        "SMA_50",
        "SMA_200",
        "volatility_30d",
        "volatility_60d",
        "volatility_180d",
        "RSI_30d",
        "RSI_90d",
        "CCI_30d",
        "ATR_30",
        "ATR_60",
        "BB_upper_30",
        "BB_lower_30",
        "Stochastic_30",
        "MACD_12_26",
        "MACD_signal_9",
        "original_value",
    ]


def short_term_forecasting(data):
    required_indicators = short_term_indicators()
    data = data[["cryptocurrency", "date"] + required_indicators].dropna()

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    }
    predictions = []
    for model_name, model in models.items():
        for crypto, crypto_data in data.groupby("cryptocurrency"):
            crypto_data = crypto_data.sort_values("date")
            X, y = prepare_data_for_ml(crypto_data)
            model.fit(X, y)
            y_pred = model.predict([X[-1]])[0]
            predictions.append(
                {
                    "cryptocurrency": crypto,
                    "predicted_change": y_pred,
                    "model_type": model_name,
                    "confidence": calculate_confidence(model, X, y),
                }
            )
    return predictions


def long_term_forecasting(data):
    required_indicators = long_term_indicators()
    data = data[["cryptocurrency", "date"] + required_indicators].dropna()

    models = {
        "Prophet": Prophet(),
        "ARIMA": ARIMA,
        "MLP": MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    }
    predictions = []
    for model_name, model in models.items():
        for crypto, crypto_data in data.groupby("cryptocurrency"):
            crypto_data = crypto_data.sort_values("date")
            X, y = prepare_data_for_ml(crypto_data)
            if model_name == "Prophet":
                df_prophet = prepare_data_for_prophet(crypto_data)
                model.fit(df_prophet)
                forecast = model.predict(df_prophet)
                y_pred = forecast["yhat"][-1]
            elif model_name == "ARIMA":
                model = model(crypto_data["value"], order=(5, 1, 0))
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=1)[0]
            elif model_name == "MLP":
                model.fit(X, y)
                y_pred = model.predict([X[-1]])[0]
            predictions.append(
                {
                    "cryptocurrency": crypto,
                    "predicted_change": y_pred,
                    "model_type": model_name,
                    "confidence": calculate_confidence(model, X, y),
                }
            )
    return predictions


def prepare_data_for_ml(data):
    if "original_value" not in data.columns:
        raise KeyError(
            f"Column 'original_value' is missing in the input data. Available columns: {list(data.columns)}"
        )

    features = data.drop(["cryptocurrency", "date", "original_value"], axis=1).iloc[:-1].values
    target = data["original_value"].shift(-1).dropna().values

    if len(features) == 0 or len(target) == 0:
        raise ValueError("Insufficient data after processing.")

    return features, target


def prepare_data_for_prophet(data):
    data["y"] = data.drop(["cryptocurrency", "date"], axis=1).mean(axis=1)
    df_prophet = data[["date", "y"]].rename(columns={"date": "ds"})

    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)

    df_prophet = df_prophet.dropna()
    return df_prophet


def calculate_confidence(model, X, y):
    if hasattr(model, "score"):
        return model.score(X, y)
    return 1.0


@transaction.atomic
def save_predictions(predictions, is_short_term=True):
    model = ShortTermCryptoPrediction if is_short_term else LongTermCryptoPrediction
    for pred in predictions:
        model.objects.update_or_create(
            cryptocurrency_pair=pred["cryptocurrency"],
            prediction_date=pd.Timestamp.now().date(),
            model_type=pred["model_type"],
            defaults={
                "predicted_price_change": pred["predicted_change"],
                "predicted_close": pred.get("predicted_close", 0),
                "confidence_level": pred["confidence"],
            },
        )


def run_forecasting():
    short_term_data = load_indicator_data(short_term_indicators())
    short_predictions = short_term_forecasting(short_term_data)
    save_predictions(short_predictions, is_short_term=True)

    long_term_data = load_indicator_data(long_term_indicators())
    long_predictions = long_term_forecasting(long_term_data)
    save_predictions(long_predictions, is_short_term=False)
