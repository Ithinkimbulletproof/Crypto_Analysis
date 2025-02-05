from prophet import Prophet
import pmdarima as pm
import pandas as pd


def train_prophet(series):
    df = series.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    print("Prophet модель обучена.")
    return model


def train_arima(series):
    model = pm.auto_arima(series, seasonal=True, m=7, trace=False)
    print("ARIMA модель обучена.")
    return model
