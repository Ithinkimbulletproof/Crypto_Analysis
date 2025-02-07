from prophet import Prophet
import pmdarima as pm
import pandas as pd
import numpy as np


def train_prophet(series, horizon="24h"):
    df = series.reset_index()
    df.columns = ["ds", "y"]

    if (df["y"] > 0).all():
        df["y"] = np.log1p(df["y"])

    model = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
    )
    model.fit(df)
    print("✅ Prophet модель обучена.")
    return model


def train_arima(series, horizon="24h"):
    series = series.dropna()
    model = pm.auto_arima(series, seasonal=True, m=7, trace=False)
    print("✅ ARIMA модель обучена.")
    return model
