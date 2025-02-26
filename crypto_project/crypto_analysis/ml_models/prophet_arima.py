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
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=0.1,
        )
        model.add_seasonality(name="daily", period=1, fourier_order=10)
        model.add_seasonality(name="weekly", period=7, fourier_order=5)
    elif horizon == "1h":
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=10,
            changepoint_range=0.8,
            changepoint_prior_scale=0.1,
        )
        model.add_seasonality(name="hourly", period=1 / 24, fourier_order=5)
    else:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            n_changepoints=5,
            changepoint_range=0.8,
            changepoint_prior_scale=0.1,
        )
        model.add_seasonality(name="daily", period=1, fourier_order=8)

    model.fit(df, iter=500)
    print("✅ Prophet модель обучена")
    return model


def train_arima(series, horizon="24h", forecast_tag=None):
    series = series.dropna()
    if len(series) > 2000:
        series = series.iloc[-2000:]

    if horizon == "24h":
        m = 7
    elif horizon == "1h":
        m = 24
    else:
        m = 7

    model = pm.auto_arima(
        series,
        seasonal=True,
        m=m,
        trace=False,
        stepwise=True,
        max_p=5,
        max_q=5,
        max_P=3,
        max_Q=3,
        max_order=10,
        error_action="ignore",
        suppress_warnings=True,
        n_jobs=-1,
    )
    print("✅ ARIMA модель обучена")
    return model
