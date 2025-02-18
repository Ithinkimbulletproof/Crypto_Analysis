from prophet import Prophet
import pmdarima as pm
import pandas as pd
import numpy as np


def train_prophet(series, horizon="24h", forecast_tag=None):
    df = series.reset_index()
    df.columns = ["ds", "y"]

    if (df["y"] > 0).all():
        df["y"] = np.log1p(df["y"])

    if horizon == "24h":
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            n_changepoints=10,
            changepoint_range=0.8,
        )
        model.add_seasonality(name="daily", period=1, fourier_order=8)
    elif horizon == "1h":
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
        )
        model.add_seasonality(name="hourly", period=1 / 24, fourier_order=3)
    else:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
        )
        model.add_seasonality(name="daily", period=1, fourier_order=8)

    model.fit(df, iter=100)
    print("✅ Prophet модель обучена")
    return model


def train_arima(series, horizon="24h", forecast_tag=None):
    series = series.dropna()
    if len(series) > 1000:
        series = series.iloc[-1000:]

    if horizon == "24h":
        m = 96
    elif horizon == "1h":
        m = 4
    else:
        m = 96

    model = pm.auto_arima(
        series,
        seasonal=True,
        m=m,
        trace=False,
        stepwise=True,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_order=5,
        error_action="ignore",
        suppress_warnings=True,
    )
    print("✅ ARIMA модель обучена")
    return model
