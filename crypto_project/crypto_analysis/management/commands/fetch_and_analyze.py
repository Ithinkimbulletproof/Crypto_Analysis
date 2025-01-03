import os
import ccxt
import logging
import datetime
import pandas as pd
from django.db.models import Max
from django.utils import timezone
from django.core.management.base import BaseCommand
from crypto_analysis.models import MarketData
from pyti.smoothed_moving_average import smoothed_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Получает данные, анализирует и обновляет базу"

    def handle(self, *args, **kwargs):
        self.fetch_data()
        self.analyze_and_update()
        self.stdout.write(
            self.style.SUCCESS("Получение, анализ и обновление завершены")
        )

    def fetch_data(self):
        exchanges = [ccxt.binance(), ccxt.bybit(), ccxt.okx()]
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        timeframe = "1h"
        default_start_date = int(
            ccxt.binance().parse8601("2021-01-01T00:00:00Z") / 1000
        )
        now = timezone.now()

        logger.info(f"Получение данных для криптовалютных пар: {symbols}")

        for exchange in exchanges:
            try:
                logger.info(f"Начало обработки биржи: {exchange.id}")
                exchange.load_markets()
                for symbol in symbols:
                    logger.info(f"Обработка пары {symbol} на {exchange.id}")
                    if symbol in exchange.markets:
                        last_date = MarketData.objects.filter(
                            cryptocurrency=symbol, exchange=exchange.id
                        ).aggregate(Max("date"))["date__max"]

                        since = (
                            int(last_date.timestamp() * 1000)
                            if last_date
                            else default_start_date
                        )

                        while True:
                            try:
                                logger.info(f"Запрос данных с {since} для {symbol}")
                                data = exchange.fetch_ohlcv(
                                    symbol, timeframe, since=since
                                )
                                if not data:
                                    logger.info(f"Данные для {symbol} завершены")
                                    break

                                for record in data:
                                    naive_date = datetime.datetime.utcfromtimestamp(
                                        record[0] / 1000
                                    )
                                    aware_date = timezone.make_aware(naive_date)

                                    MarketData.objects.update_or_create(
                                        cryptocurrency=symbol,
                                        date=aware_date,
                                        exchange=exchange.id,
                                        defaults={
                                            "open_price": record[1],
                                            "high_price": record[2],
                                            "low_price": record[3],
                                            "close_price": record[4],
                                            "volume": record[5],
                                        },
                                    )
                                    logger.info(f"Добавлена запись: {record}")

                                since = data[-1][0] + 1
                            except Exception as e:
                                logger.error(
                                    f"Ошибка загрузки данных {symbol} для {exchange.id}: {str(e)}"
                                )
                                break
            except Exception as e:
                logger.error(f"Ошибка работы с биржей {exchange.id}: {str(e)}")
        logger.info("Загрузка данных завершена")

    def analyze_and_update(self):
        cryptocurrencies = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        models = {
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            ),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        total_predictions = 0
        correct_predictions = 0

        base_output_dir = "data/analyzed"
        os.makedirs(base_output_dir, exist_ok=True)

        for crypto in cryptocurrencies:
            try:
                logger.info(f"Обработка данных для {crypto}")
                data_all = MarketData.objects.filter(cryptocurrency=crypto).order_by(
                    "-date"
                )
                if not data_all.exists():
                    logger.warning(f"Нет данных для {crypto}. Пропускаем.")
                    continue

                df = pd.DataFrame(list(data_all.values()))
                if len(df) < 14:
                    logger.warning(f"Недостаточно данных для {crypto}. Пропускаем.")
                    continue

                logger.info(f"Количество записей в DataFrame до обработки: {len(df)}")

                close_prices = df["close_price"].tolist()
                df["SMA_14"] = (
                    sma(close_prices, 14)
                    if len(close_prices) >= 14
                    else [None] * len(df)
                )
                df["RSI_14"] = (
                    rsi(close_prices, 14)
                    if len(close_prices) >= 14
                    else [None] * len(df)
                )
                df["volatility"] = df["close_price"].rolling(window=14).std()
                df["volume_change"] = df["volume"].pct_change()
                df["price_change"] = df["close_price"].pct_change().shift(-1)
                df["MACD"] = (
                    df["close_price"].ewm(span=12).mean()
                    - df["close_price"].ewm(span=26).mean()
                )
                df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
                df["hour_of_day"] = pd.to_datetime(df["date"]).dt.hour
                df["actual_change"] = (df["price_change"] > 0).astype(int)

                logger.info(f"NaN в столбцах перед обработкой:\n{df.isnull().sum()}")
                df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
                df.dropna(
                    subset=["SMA_14", "RSI_14", "volatility", "volume_change", "MACD"],
                    inplace=True,
                )

                if df.empty:
                    logger.warning(
                        f"Недостаточно данных для анализа {crypto}. Пропускаем."
                    )
                    continue

                feature_columns = [
                    "close_price",
                    "volume",
                    "SMA_14",
                    "RSI_14",
                    "volatility",
                    "volume_change",
                    "MACD",
                    "day_of_week",
                    "hour_of_day",
                ]
                X = df[feature_columns].copy()
                y = df["actual_change"]

                scaler = StandardScaler()
                smote = SMOTE(random_state=42)

                X.fillna(X.mean(), inplace=True)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                X_scaled = scaler.fit_transform(X_resampled)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled,
                    y_resampled,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_resampled,
                )

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions)
                    logger.info(
                        f"Точность модели {model_name}: {accuracy:.2%}, F1-score: {f1:.2%}"
                    )

                    total_predictions += len(predictions)
                    correct_predictions += sum(predictions == y_test)

            except Exception as e:
                logger.error(f"Ошибка обработки {crypto}: {str(e)}")

        overall_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        logger.info(f"Общая точность моделей: {overall_accuracy:.2%}")
