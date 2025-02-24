import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error


def prepare_features(df):
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
        df = df[df["date"] >= five_years_ago]
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek

    cols_to_drop = ["cryptocurrency", "date", "news_hour", "date_y", "entities"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

    if "close_price" not in df.columns:
        raise KeyError(
            f"Колонка 'close_price' отсутствует в DataFrame. Доступные колонки: {df.columns.tolist()}"
        )

    df["lag_1h"] = df["close_price"].shift(4)
    df["lag_24h"] = df["close_price"].shift(96)

    df = df.dropna()
    return df


def train_model(X, y, model_type, params):
    if not params:
        raise ValueError("Параметры модели должны быть переданы извне.")

    split_index = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    if model_type.lower() == "xgb":
        model = xgb.XGBRegressor(**params)
    elif model_type.lower() == "lgb":
        model = lgb.LGBMRegressor(**params)
    else:
        raise ValueError("model_type должен быть 'xgb' или 'lgb'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"{model_type.upper()} validation MSE: {mse:.4f}")
    return model


def train_xgboost_and_lightgbm(df, params, framework, target):
    df_prepared = prepare_features(df)

    df_prepared["price_1h"] = df_prepared["lag_1h"]
    df_prepared["price_24h"] = df_prepared["lag_24h"]

    X = df_prepared.drop(columns=["close_price", "price_1h", "price_24h"])

    if target:
        y = df_prepared[target]
        horizon = "1 час" if target == "price_1h" else "24 часа"
        print(f"Обучение {framework.upper()} для предсказания цены через {horizon}:")
        return train_model(X, y, model_type=framework, params=params)

    y_1h = df_prepared["price_1h"]
    y_24h = df_prepared["price_24h"]
    models = {}

    for timeframe, y in [("1h", y_1h), ("24h", y_24h)]:
        print(f"Обучение {framework.upper()} для предсказания цены через {timeframe}:")
        models[f"{framework}_{timeframe}"] = train_model(
            X, y, model_type=framework, params=params
        )

    return models
