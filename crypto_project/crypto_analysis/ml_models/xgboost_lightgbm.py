import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_features(df):
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"], errors='coerce')
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Индекс не удалось преобразовать в DatetimeIndex.")
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["lag_1h"] = df["close_price"].shift(1)
    df["lag_24h"] = df["close_price"].shift(24)
    df = df.dropna()
    return df

def train_model(X, y, model_type="xgb", random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    if model_type.lower() == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
        )
    elif model_type.lower() == "lgb":
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=random_state
        )
    else:
        raise ValueError("model_type must be either 'xgb' or 'lgb'")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"{model_type.upper()} validation MSE: {mse:.4f}")
    return model

def train_xgboost_and_lightgbm(df):
    df = prepare_features(df)
    df["price_1h"] = df["lag_1h"]
    df["price_24h"] = df["lag_24h"]
    X = df.drop(columns=["close_price", "price_1h", "price_24h"])
    y_1h = df["price_1h"]
    y_24h = df["price_24h"]
    model_xgb_1h = train_model(X, y_1h, "xgb")
    model_xgb_24h = train_model(X, y_24h, "xgb")
    model_lgb_1h = train_model(X, y_1h, "lgb")
    model_lgb_24h = train_model(X, y_24h, "lgb")
    return {
        "xgb_1h": model_xgb_1h,
        "xgb_24h": model_xgb_24h,
        "lgb_1h": model_lgb_1h,
        "lgb_24h": model_lgb_24h,
    }
