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


def train_model(X, y, model_type="xgb", params=None, random_state=42):
    split_index = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    if params is None:
        params = {}

    if model_type.lower() == "xgb":
        default_params = {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": random_state,
        }
        default_params.update(params)
        model = xgb.XGBRegressor(**default_params)
    elif model_type.lower() == "lgb":
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 64,
            "random_state": random_state,
        }
        default_params.update(params)
        model = lgb.LGBMRegressor(**default_params)
    else:
        raise ValueError("model_type must be either 'xgb' or 'lgb'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"{model_type.upper()} validation MSE: {mse:.4f}")
    return model


def train_xgboost_and_lightgbm(df, params=None, framework=None, target=None):
    df_prepared = prepare_features(df)

    df_prepared["price_1h"] = df_prepared["lag_1h"]
    df_prepared["price_24h"] = df_prepared["lag_24h"]

    X = df_prepared.drop(columns=["close_price", "price_1h", "price_24h"])

    if target is not None:
        y = df_prepared[target]
        horizon = "1 час" if target == "price_1h" else "24 часа"
        if framework == "xgboost":
            print(f"Обучение XGBoost для предсказания цены через {horizon}:")
            model = train_model(X, y, model_type="xgb", params=params)
            return model
        elif framework == "lightgbm":
            print(f"Обучение LightGBM для предсказания цены через {horizon}:")
            model = train_model(X, y, model_type="lgb", params=params)
            return model
        else:
            print(f"Обучение XGBoost для предсказания цены через {horizon}:")
            model = train_model(X, y, model_type="xgb", params=params)
            return model
    else:
        y_1h = df_prepared["price_1h"]
        y_24h = df_prepared["price_24h"]
        if framework == "xgboost":
            print("Обучение XGBoost для предсказания цены через 1 час:")
            model_1h = train_model(X, y_1h, model_type="xgb", params=params)
            print("Обучение XGBoost для предсказания цены через 24 часа:")
            model_24h = train_model(X, y_24h, model_type="xgb", params=params)
            return {
                "xgb_1h": model_1h,
                "xgb_24h": model_24h,
            }
        elif framework == "lightgbm":
            print("Обучение LightGBM для предсказания цены через 1 час:")
            model_1h = train_model(X, y_1h, model_type="lgb", params=params)
            print("Обучение LightGBM для предсказания цены через 24 часа:")
            model_24h = train_model(X, y_24h, model_type="lgb", params=params)
            return {
                "lgb_1h": model_1h,
                "lgb_24h": model_24h,
            }
        else:
            print("Обучение XGBoost для предсказания цены через 1 час:")
            model_xgb_1h = train_model(X, y_1h, model_type="xgb", params=params)
            print("Обучение XGBoost для предсказания цены через 24 часа:")
            model_xgb_24h = train_model(X, y_24h, model_type="xgb", params=params)
            print("Обучение LightGBM для предсказания цены через 1 час:")
            model_lgb_1h = train_model(X, y_1h, model_type="lgb", params=params)
            print("Обучение LightGBM для предсказания цены через 24 часа:")
            model_lgb_24h = train_model(X, y_24h, model_type="lgb", params=params)
            return {
                "xgb_1h": model_xgb_1h,
                "xgb_24h": model_xgb_24h,
                "lgb_1h": model_lgb_1h,
                "lgb_24h": model_lgb_24h,
            }
