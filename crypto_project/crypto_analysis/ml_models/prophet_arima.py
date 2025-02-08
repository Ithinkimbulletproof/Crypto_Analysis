from prophet import Prophet
import pmdarima as pm
import pandas as pd
import numpy as np


def train_prophet(series, horizon="24h"):
    df = series.reset_index()
    df.columns = ["ds", "y"]

    if (df["y"] > 0).all():
        df["y"] = np.log1p(df["y"])

    if horizon == "24h":
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
        )
    elif horizon == "1h":
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
        )
        model.add_seasonality(name="daily", period=24, fourier_order=3)
    else:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
        )

    model.fit(df, iter=200, n_jobs=2)
    print("✅ Prophet модель обучена для горизонта:", horizon)
    return model


def train_arima(series, horizon="24h"):
    series = series.dropna()
    model = pm.auto_arima(series, seasonal=True, m=7, trace=False, n_jobs=2)
    print("✅ ARIMA модель обучена.")
    return model
