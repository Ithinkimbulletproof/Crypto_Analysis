import logging
import numpy as np
import pandas as pd
from django.utils import timezone
from xgboost import XGBRegressor
from django.db import transaction
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from crypto_analysis.models import (
    IndicatorData,
    ShortTermCryptoPrediction,
    MarketData,
    NewsArticle,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_market_and_indicator_data(selected_indicators=None):
    try:
        now = timezone.now()

        market_data = MarketData.objects.filter(
            date__gte=now - timezone.timedelta(days=3 * 365)
        ).values(
            "cryptocurrency",
            "date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "exchange",
        )
        df_market = pd.DataFrame(market_data)
        df_market["date"] = pd.to_datetime(df_market["date"], utc=True).dt.tz_localize(
            None
        )

        indicator_query = IndicatorData.objects.filter(
            date__gte=now - timezone.timedelta(days=2 * 365)
        )
        if selected_indicators:
            indicator_query = indicator_query.filter(
                indicator_name__in=selected_indicators
            )

        indicator_data = indicator_query.values(
            "cryptocurrency", "date", "indicator_name", "value"
        )

        df_indicators = pd.DataFrame(indicator_data)
        df_indicators["date"] = pd.to_datetime(
            df_indicators["date"], utc=True
        ).dt.tz_localize(None)
        df_indicators_wide = df_indicators.pivot_table(
            index=["cryptocurrency", "date"],
            columns="indicator_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        news_data = NewsArticle.objects.filter(
            published_at__gte=now - timezone.timedelta(days=3)
        ).values("published_at", "title", "description", "sentiment", "polarity")

        df_news = pd.DataFrame(news_data)
        df_news.rename(columns={"published_at": "date"}, inplace=True)
        df_news["date"] = (
            pd.to_datetime(df_news["date"], utc=True).dt.tz_localize(None).dt.floor("h")
        )

        sentiment_encoder = LabelEncoder()
        df_news["sentiment"] = sentiment_encoder.fit_transform(df_news["sentiment"])
        df_news["polarity"] = pd.to_numeric(df_news["polarity"], errors="coerce")

        df_combined = pd.merge(
            df_indicators_wide, df_market, on=["cryptocurrency", "date"], how="inner"
        )

        df_combined = pd.merge_asof(
            df_combined.sort_values("date"),
            df_news.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta("6h"),
        )

        numeric_cols = df_combined.select_dtypes(include=np.number).columns
        df_combined[numeric_cols] = df_combined[numeric_cols].ffill().fillna(0)

        text_cols = ["title", "description"]
        df_combined[text_cols] = df_combined[text_cols].fillna("")

        df_combined["sentiment"] = df_combined["sentiment"].fillna(0)
        df_combined["polarity"] = df_combined["polarity"].fillna(0)

        if df_combined.isnull().any().any():
            logger.warning(
                f"Пропущенные значения после обработки: {df_combined.isnull().sum()}"
            )

        return df_combined

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}", exc_info=True)
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
    ]


def prepare_data_for_ml(data):
    if data.empty:
        raise ValueError("Пустой DataFrame на входе")

    if "close_price" not in data.columns:
        raise KeyError(
            f"Отсутствует столбец close_price в данных. Доступные колонки: {list(data.columns)}"
        )

    if len(data) < 3:
        raise ValueError(f"Недостаточно данных для обучения: {len(data)} samples")

    data_clean = data.dropna(subset=["close_price"]).copy()

    features = data_clean.drop(
        columns=["cryptocurrency", "date", "close_price"], errors="ignore"
    )
    features = features.fillna(0)
    target = data_clean["close_price"].shift(-1).dropna()

    features = features.iloc[:-1]
    target = target.iloc[:-1] if len(target) > len(features) else target

    features_array = features.copy().values
    target_array = target.copy().values

    if len(features_array) != len(target_array):
        min_length = min(len(features_array), len(target_array))
        features_array = features_array[:min_length]
        target_array = target_array[:min_length]

    if len(features_array) == 0 or len(target_array) == 0:
        raise ValueError("Нулевой размер данных после обработки")

    if np.isnan(features_array).any():
        logger.warning("Найдены NaN в фичах, заменяем на 0")
        features_array = np.nan_to_num(features_array)

    if np.isnan(target_array).any():
        logger.warning("Найдены NaN в целевой переменной, заменяем на 0")
        target_array = np.nan_to_num(target_array)

    return features_array, target_array


def short_term_forecasting(data):
    if data.empty:
        logger.error("Пустые входные данные для прогнозирования")
        return []

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=50, max_depth=3, min_samples_split=3, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.05, enable_categorical=True
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=2,
            num_leaves=7,
            force_col_wise=True,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=50, depth=3, learning_rate=0.05, verbose=0
        ),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.05),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(50,),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            tol=1e-5,
            n_iter_no_change=20,
        ),
    }

    features_config = {
        "RandomForest": ["price_change_1d", "SMA_7", "RSI_7d", "volume"],
        "XGBoost": ["price_change_7d", "SMA_7", "RSI_14d", "volume"],
        "LightGBM": ["SMA_14", "volatility_14d", "RSI_30d", "volume"],
        "CatBoost": ["SMA_30", "RSI_14d", "volume"],
        "AdaBoost": ["price_change_14d", "ATR_14", "volume"],
        "MLP": ["price_change_1d", "RSI_7d", "SMA_7", "volume"],
    }

    predictions = []

    for crypto, crypto_data in data.groupby("cryptocurrency"):
        crypto_data = crypto_data.sort_values("date").reset_index(drop=True)

        if len(crypto_data) < 5:
            logger.warning(
                f"Недостаточно данных для {crypto}: {len(crypto_data)} записей"
            )
            continue

        current_price = crypto_data["close_price"].iloc[-1]

        for model_name, model in models.items():
            try:
                selected_features = features_config[model_name]
                available_features = [
                    f for f in selected_features if f in crypto_data.columns
                ]

                missing_features = set(selected_features) - set(available_features)
                if missing_features:
                    logger.warning(
                        f"Отсутствуют фичи для {model_name} ({crypto}): {missing_features}"
                    )

                if not available_features:
                    logger.warning(f"Нет доступных фич для {model_name} в {crypto}")
                    continue

                try:
                    X, y = prepare_data_for_ml(
                        crypto_data[
                            ["cryptocurrency", "date", "close_price"]
                            + available_features
                        ]
                    )
                except Exception as e:
                    logger.error(f"Ошибка подготовки данных для {crypto}: {str(e)}")
                    continue

                split_idx = max(2, int(0.8 * len(X)))
                if split_idx < 2 or (len(X) - split_idx) < 1:
                    logger.warning(
                        f"Недостаточно данных для разделения: {len(X)} samples"
                    )
                    X_train, y_train = X.copy(), y.copy()
                    X_test, y_test = np.array([]), np.array([])
                else:
                    X_train, X_test = X[:split_idx].copy(), X[split_idx:].copy()
                    y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()

                model.fit(X_train, y_train)

                test_score = 0.0
                if len(X_test) > 0:
                    try:
                        test_score = model.score(X_test, y_test)
                    except Exception as e:
                        logger.warning(f"Ошибка оценки модели: {str(e)}")

                last_features = (
                    crypto_data[available_features].iloc[-1].values.reshape(1, -1)
                )
                y_pred = model.predict(last_features)[0]

                price_change = y_pred - current_price

                predictions.append(
                    {
                        "cryptocurrency": crypto,
                        "predicted_change": price_change,
                        "predicted_close": y_pred,
                        "current_price": current_price,
                        "model_type": model_name,
                        "confidence": max(0.0, min(test_score, 1.0)),
                    }
                )

            except Exception as e:
                logger.error(
                    f"Ошибка в модели {model_name} для {crypto}: {str(e)}",
                    exc_info=True,
                )
                continue

    return predictions


@transaction.atomic
def save_predictions(predictions, is_short_term=True):
    if not predictions:
        logger.warning("Пустой список прогнозов для сохранения")
        return

    current_date = timezone.now().date()

    for pred in predictions:
        try:
            ShortTermCryptoPrediction.objects.update_or_create(
                cryptocurrency_pair=pred["cryptocurrency"],
                prediction_date=current_date,
                model_type=pred["model_type"],
                defaults={
                    "predicted_price_change": pred["predicted_change"],
                    "predicted_close": pred["predicted_close"],
                    "current_price": pred["current_price"],
                    "confidence_level": pred["confidence"],
                },
            )
        except Exception as e:
            logger.error(
                f"Ошибка сохранения прогноза для {pred['cryptocurrency']}: {str(e)}"
            )
            continue


def run_short_predictions():
    try:
        indicators = short_term_indicators()
        data = load_market_and_indicator_data(indicators)

        if not data.empty:
            logger.info(f"Начало прогнозирования на {len(data)} записях")
            predictions = short_term_forecasting(data)

            if predictions:
                save_predictions(predictions, is_short_term=True)
                logger.info(f"Успешно сохранено {len(predictions)} прогнозов")
            else:
                logger.warning("Нет прогнозов для сохранения")
        else:
            logger.error("Нет данных для прогнозирования")

    except Exception as e:
        logger.error(
            f"Критическая ошибка в run_short_predictions: {str(e)}", exc_info=True
        )
