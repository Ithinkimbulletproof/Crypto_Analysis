import logging
import numpy as np
import pandas as pd
from django.utils import timezone
from django.db import transaction
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from crypto_analysis.models import (
    ShortTermCryptoPrediction,
    MarketData,
    NewsArticle,
    IndicatorData,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    @staticmethod
    def load_data(selected_indicators=None):
        try:
            now = timezone.now()
            three_years_ago = now - timezone.timedelta(days=3 * 365)

            market_data = MarketData.objects.filter(
                date__gte=three_years_ago
            ).values_list(
                "cryptocurrency",
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "exchange",
                named=True,
            )
            df_market = pd.DataFrame(
                market_data,
                columns=[
                    "cryptocurrency",
                    "date",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "volume",
                    "exchange",
                ],
            )
            df_market["date"] = pd.to_datetime(
                df_market["date"], utc=True
            ).dt.tz_localize(None)

            indicator_query = (
                IndicatorData.objects.filter(
                    date__gte=three_years_ago, indicator_name__in=selected_indicators
                )
                if selected_indicators
                else IndicatorData.objects.filter(date__gte=three_years_ago)
            )

            indicators = indicator_query.values_list(
                "cryptocurrency", "date", "indicator_name", "value", named=True
            )
            df_indicators = pd.DataFrame(
                indicators,
                columns=["cryptocurrency", "date", "indicator_name", "value"],
            )
            df_indicators["date"] = pd.to_datetime(
                df_indicators["date"], utc=True
            ).dt.tz_localize(None)

            df_indicators_wide = df_indicators.pivot_table(
                index=["cryptocurrency", "date"],
                columns="indicator_name",
                values="value",
                aggfunc="mean",
            ).reset_index()

            news_data = NewsArticle.objects.filter(
                published_at__gte=now - timezone.timedelta(days=3)
            ).values_list(
                "published_at",
                "title",
                "description",
                "sentiment",
                "polarity",
                "cryptocurrency",
                named=True,
            )
            df_news = pd.DataFrame(
                news_data,
                columns=[
                    "date",
                    "title",
                    "description",
                    "sentiment",
                    "polarity",
                    "cryptocurrency",
                ],
            )
            df_news["date"] = (
                pd.to_datetime(df_news["date"], utc=True)
                .dt.tz_localize(None)
                .dt.floor("h")
            )
            df_news["sentiment"] = df_news["sentiment"].astype("category").cat.codes
            df_news["polarity"] = pd.to_numeric(
                df_news["polarity"], errors="coerce"
            ).fillna(0)

            df_combined = pd.merge(
                df_market, df_indicators_wide, on=["cryptocurrency", "date"], how="left"
            )

            df_combined = pd.merge_asof(
                df_combined.sort_values("date"),
                df_news.sort_values("date"),
                on="date",
                by="cryptocurrency",
                direction="nearest",
                tolerance=pd.Timedelta("6h"),
            )

            numeric_cols = df_combined.select_dtypes(include=np.number).columns
            df_combined[numeric_cols] = (
                df_combined[numeric_cols].interpolate(limit_direction="both").fillna(0)
            )

            df_combined = df_combined.drop_duplicates(subset=["cryptocurrency", "date"])

            logger.info(f"Data loaded successfully. Shape: {df_combined.shape}")
            return df_combined

        except Exception as e:
            logger.error(f"Data loading error: {str(e)}", exc_info=True)
            return pd.DataFrame()


class FeatureEngineer:
    @staticmethod
    def prepare_features(data, lookback_window=30):
        if data.empty:
            raise ValueError("Empty input data")

        data = data.sort_values("date").set_index("date")

        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month
        data["hour"] = data.index.hour

        for lag in range(1, lookback_window + 1):
            data[f"close_lag_{lag}"] = data["close_price"].shift(lag)

        data["price_range"] = data["high_price"] - data["low_price"]
        data["typical_price"] = (
            data["high_price"] + data["low_price"] + data["close_price"]
        ) / 3

        data["news_length"] = data["description"].str.len().fillna(0)

        data = data.ffill().bfill().dropna(subset=["close_price"])

        return data


class ModelFactory:
    @staticmethod
    def get_models():
        return {
            "RandomForest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 5],
                },
            },
            "XGBoost": {
                "model": XGBRegressor(enable_categorical=True, random_state=42),
                "params": {
                    "n_estimators": [100, 150],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                },
            },
            "LightGBM": {
                "model": LGBMRegressor(random_state=42),
                "params": {
                    "num_leaves": [31, 63],
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200],
                },
            },
        }


class CryptoPredictor:
    def __init__(self, lookback_window=30):
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()

    def train_predict(self, data):
        predictions = []

        for crypto, crypto_data in data.groupby("cryptocurrency"):
            try:
                crypto_data = FeatureEngineer.prepare_features(
                    crypto_data, self.lookback_window
                )
                features = crypto_data.drop(
                    columns=["cryptocurrency", "close_price"], errors="ignore"
                )
                target = crypto_data["close_price"]

                if len(features) < 100:
                    logger.warning(
                        f"Not enough data for {crypto}: {len(features)} samples"
                    )
                    continue

                split_idx = int(0.8 * len(features))
                X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = (
                    self.scaler.transform(X_test) if not X_test.empty else None
                )

                model_results = []
                for name, config in ModelFactory.get_models().items():
                    try:
                        grid_search = GridSearchCV(
                            estimator=config["model"],
                            param_grid=config["params"],
                            cv=TimeSeriesSplit(n_splits=3),
                            scoring="neg_mean_absolute_percentage_error",
                            n_jobs=-1,
                            verbose=0,
                        )
                        grid_search.fit(X_train_scaled, y_train)

                        best_model = grid_search.best_estimator_
                        test_score = (
                            mean_absolute_percentage_error(
                                y_test, best_model.predict(X_test_scaled)
                            )
                            if not X_test.empty
                            else 0
                        )

                        model_results.append(
                            {
                                "name": name,
                                "model": best_model,
                                "train_score": grid_search.best_score_,
                                "test_score": test_score,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Model {name} failed for {crypto}: {str(e)}")

                current_features = self.scaler.transform(features.iloc[[-1]])
                current_price = target.iloc[-1]

                for result in model_results:
                    try:
                        predicted_price = result["model"].predict(current_features)[0]
                        predictions.append(
                            {
                                "cryptocurrency": crypto,
                                "model_type": result["name"],
                                "current_price": current_price,
                                "predicted_close": predicted_price,
                                "predicted_change": predicted_price - current_price,
                                "confidence": max(0, 1 - abs(result["test_score"])),
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Prediction failed for {result['name']} on {crypto}: {str(e)}"
                        )

            except Exception as e:
                logger.error(f"Processing failed for {crypto}: {str(e)}", exc_info=True)

        return predictions


@transaction.atomic
def save_predictions(predictions):
    if not predictions:
        logger.warning("No predictions to save")
        return

    current_date = timezone.now().date()
    bulk_entries = []

    for pred in predictions:
        bulk_entries.append(
            ShortTermCryptoPrediction(
                cryptocurrency_pair=pred["cryptocurrency"],
                prediction_date=current_date,
                model_type=pred["model_type"],
                predicted_price_change=pred["predicted_change"],
                predicted_close=pred["predicted_close"],
                current_price=pred["current_price"],
                confidence_level=pred["confidence"],
            )
        )

    try:
        ShortTermCryptoPrediction.objects.bulk_create(
            bulk_entries,
            update_conflicts=True,
            update_fields=[
                "predicted_price_change",
                "predicted_close",
                "current_price",
                "confidence_level",
            ],
            unique_fields=["cryptocurrency_pair", "prediction_date", "model_type"],
        )
        logger.info(f"Successfully saved/updated {len(bulk_entries)} predictions")
    except Exception as e:
        logger.error(f"Failed to save predictions: {str(e)}")


def run_predictions():
    try:
        logger.info("Starting prediction pipeline")

        data = DataLoader.load_data(short_term_indicators())
        if data.empty:
            raise ValueError("No data available for prediction")

        predictor = CryptoPredictor(lookback_window=30)
        predictions = predictor.train_predict(data)

        if predictions:
            save_predictions(predictions)
            logger.info(
                f"Prediction pipeline completed. Generated {len(predictions)} predictions"
            )
        else:
            logger.warning("Prediction pipeline completed with no results")

        return True

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}", exc_info=True)
        return False


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
        "BB_upper_14",
        "BB_lower_14",
        "BB_upper_30",
        "BB_lower_30",
        "MACD_12_26",
        "MACD_signal_9",
        "Lag_12",
        "Lag_26",
        "Lag_9",
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
