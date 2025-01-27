import logging
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from django.db import transaction
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from crypto_analysis.models import IndicatorData, ShortTermCryptoPrediction, MarketData, NewsArticle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_market_and_indicator_data(selected_indicators=None):
    try:
        query = IndicatorData.objects.all()
        if selected_indicators:
            query = query.filter(indicator_name__in=selected_indicators)
        indicators = query.values("cryptocurrency", "date", "indicator_name", "value")
        df_indicators = pd.DataFrame(indicators)
        logger.info(f"Загружено индикаторов: {df_indicators.shape}")

        df_indicators["date"] = pd.to_datetime(df_indicators["date"], utc=True).dt.tz_localize(None)

        df_indicators_wide = df_indicators.pivot(
            index=["cryptocurrency", "date"], columns="indicator_name", values="value"
        ).reset_index()
        logger.info(f"Преобразование индикаторов в широкий формат: {df_indicators_wide.shape}")

        cryptos = df_indicators_wide["cryptocurrency"].unique()
        market_data = (
            MarketData.objects.filter(cryptocurrency__in=cryptos)
            .order_by("date")
            .values(
                "cryptocurrency",
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "exchange",
            )
        )
        df_market = pd.DataFrame(market_data)
        logger.info(f"Загружено рыночных данных: {df_market.shape}")

        df_market["date"] = pd.to_datetime(df_market["date"], utc=True).dt.tz_localize(None)

        news_data = NewsArticle.objects.filter(published_at__isnull=False).values(
            "published_at", "title", "description", "sentiment", "polarity"
        )
        df_news = pd.DataFrame(news_data)
        logger.info(f"Загружено новостных данных: {df_news.shape}")

        df_news.rename(columns={"published_at": "date"}, inplace=True)
        df_news["date"] = pd.to_datetime(df_news["date"], utc=True).dt.tz_localize(None)

        df_indicators["date"] = df_indicators["date"].dt.floor('h')
        df_market["date"] = df_market["date"].dt.floor('h')
        df_news["date"] = df_news["date"].dt.floor('h')

        logger.info("Начинаем объединение данных...")

        df_combined = pd.merge(
            df_indicators_wide, df_market, on=["cryptocurrency", "date"], how="left"
        )
        logger.info(f"После объединения индикаторов и рыночных данных: {df_combined.shape}")

        df_combined = pd.merge_asof(
            df_combined.sort_values("date"),
            df_news.sort_values("date"),
            on="date",
            direction="nearest"
        )
        logger.info(f"После объединения с новостями: {df_combined.shape}")

        if df_combined.isnull().any(axis=1).any():
            logger.warning(f"Обнаружены строки с пропущенными значениями: {df_combined.isnull().sum()}")

        return df_combined

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()


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
        "seasonality_weekday_Wednesday",
        "seasonality_weekday_Tuesday",
        "seasonality_weekday_Thursday",
        "seasonality_weekday_Sunday",
        "seasonality_weekday_Saturday",
        "seasonality_weekday_Monday",
        "seasonality_weekday_Friday",
        "seasonality_month_1",
        "seasonality_month_2",
        "seasonality_month_3",
        "seasonality_month_4",
        "seasonality_month_5",
        "seasonality_month_6",
        "seasonality_month_7",
        "seasonality_month_8",
        "seasonality_month_9",
        "seasonality_month_10",
        "seasonality_month_11",
        "seasonality_month_12",
        "BTC_Correlation",
        "ETH_Correlation",
        "value",
        "volume",
        "sentiment",
        "polarity",
        "title",
        "description",
    ]


def short_term_forecasting(data):
    required_indicators = short_term_indicators()
    required_indicators = [
        indicator for indicator in required_indicators if indicator != "value"
    ]

    data = data[["cryptocurrency", "date"] + required_indicators + ["close_price"]].dropna()

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, num_leaves=31),
        "CatBoost": CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, verbose=0),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=50, activation="relu", solver="adam", learning_rate="adaptive", max_iter=100, random_state=42)
    }

    features = {
        "RandomForest": [
            "price_change_1d", "SMA_7", "SMA_14", "RSI_7d", "volatility_7d", "MACD_12_26",
            "seasonality_weekday_Monday", "BTC_Correlation", "ETH_Correlation", "volume"
        ],
        "XGBoost": [
            "price_change_7d", "SMA_7", "RSI_14d", "ATR_14", "Stochastic_7", "seasonality_weekday_Thursday",
            "BTC_Correlation", "sentiment", "volume"
        ],
        "LightGBM": [
            "SMA_14", "volatility_14d", "RSI_30d", "Stochastic_14", "seasonality_month_1",
            "ETH_Correlation", "polarity", "volume"
        ],
        "CatBoost": [
            "SMA_30", "RSI_14d", "volatility_30d", "MACD_signal_9", "seasonality_weekday_Saturday",
            "BTC_Correlation", "sentiment", "title", "volume"
        ],
        "AdaBoost": [
            "price_change_14d", "ATR_14", "seasonality_weekday_Friday", "ETH_Correlation",
            "description", "volume"
        ],
        "MLP": [
            "price_change_1d", "RSI_7d", "RSI_14d", "SMA_7", "SMA_14", "MACD_12_26",
            "seasonality_weekday_Monday", "BTC_Correlation", "ETH_Correlation", "sentiment",
            "polarity", "volume"
        ]
    }

    predictions = []
    for model_name, model in models.items():
        for crypto, crypto_data in data.groupby("cryptocurrency"):
            crypto_data = crypto_data.sort_values("date")

            selected_features = features[model_name]

            X = crypto_data[selected_features]
            y = crypto_data["close_price"]

            model.fit(X, y)
            y_pred = model.predict([X.iloc[-1]])[0]

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
    if "close_price" not in data.columns:
        raise KeyError(
            f"Column 'close_price' is missing in the input data. Available columns: {list(data.columns)}"
        )

    features = data.drop(["cryptocurrency", "date"], axis=1).iloc[:-1].values
    target = data["close_price"].shift(-1).dropna().values

    if len(features) == 0 or len(target) == 0:
        raise ValueError("Insufficient data after processing.")

    return features, target


def prepare_data_for_prophet(data):
    data = data.copy()
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
    model = ShortTermCryptoPrediction
    for pred in predictions:
        model.objects.update_or_create(
            cryptocurrency_pair=pred["cryptocurrency"],
            prediction_date=pd.Timestamp.now().date(),
            model_type=pred["model_type"],
            defaults={
                "predicted_price_change": pred["predicted_change"],
                "predicted_close": pred["predicted_change"],
                "confidence_level": pred["confidence"],
            },
        )


def run_short_predictions():
    short_term_data = load_market_and_indicator_data(short_term_indicators())
    short_predictions = short_term_forecasting(short_term_data)
    save_predictions(short_predictions, is_short_term=True)
