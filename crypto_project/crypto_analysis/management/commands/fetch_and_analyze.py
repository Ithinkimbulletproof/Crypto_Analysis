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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
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
        default_start_date = self.get_default_start_date()
        now = timezone.now()

        logger.info(f"Получение данных для криптовалютных пар: {symbols}")
        logger.info(f"Стартовый timestamp для данных: {default_start_date}")

        for exchange in exchanges:
            logger.info(f"Начало обработки биржи: {exchange.id}")
            try:
                self.process_exchange(exchange, symbols, default_start_date)
                logger.info(f"Завершена обработка биржи: {exchange.id}")
            except Exception as e:
                logger.error(f"Ошибка работы с биржей {exchange.id}: {str(e)}")

        logger.info("Загрузка данных завершена")

    def get_default_start_date(self):
        logger.info("Получение даты начала для загрузки данных.")
        start_date = int(ccxt.binance().parse8601("2021-01-01T00:00:00Z") / 1000)
        logger.info(f"Дата начала для загрузки данных: {start_date}")
        return start_date

    def process_exchange(self, exchange, symbols, default_start_date):
        logger.info(f"Начало обработки биржи: {exchange.id}")
        try:
            exchange.load_markets()
            for symbol in symbols:
                logger.info(f"Обработка пары {symbol} на {exchange.id}")
                if symbol in exchange.markets:
                    last_date = self.get_last_date(symbol, exchange)
                    logger.info(
                        f"Последняя дата для {symbol} на {exchange.id}: {last_date}"
                    )
                    since = self.get_since(last_date, default_start_date)
                    logger.info(
                        f"Дата начала для загрузки данных для {symbol} на {exchange.id}: {since}"
                    )
                    self.fetch_and_store_data(exchange, symbol, since)
                else:
                    logger.warning(f"Пара {symbol} не найдена на {exchange.id}")
        except Exception as e:
            logger.error(f"Ошибка при обработке биржи {exchange.id}: {str(e)}")

    def get_last_date(self, symbol, exchange):
        try:
            last_date = MarketData.objects.filter(
                cryptocurrency=symbol, exchange=exchange.id
            ).aggregate(Max("date"))["date__max"]
            logger.info(f"Последняя дата для {symbol} на {exchange.id}: {last_date}")
            return last_date
        except Exception as e:
            logger.error(
                f"Ошибка при получении последней даты для {symbol} на {exchange.id}: {str(e)}"
            )
            return None

    def get_since(self, last_date, default_start_date):
        if last_date:
            since = int(last_date.timestamp() * 1000)
            logger.info(f"Использована последняя дата: {since}")
        else:
            since = default_start_date
            logger.info(
                f"Последняя дата не найдена, используется дата по умолчанию: {since}"
            )

        return since

    def fetch_and_store_data(self, exchange, symbol, since):
        while True:
            try:
                logger.info(f"Запрос данных с {since} для {symbol} на {exchange.id}")
                data = exchange.fetch_ohlcv(symbol, "1h", since=since)
                if not data:
                    logger.info(f"Данные для {symbol} завершены на {exchange.id}")
                    break

                for record in data:
                    self.store_data(symbol, exchange, record)

                since = data[-1][0] + 1
                logger.info(
                    f"Обновлена дата запроса на {since} для {symbol} на {exchange.id}"
                )
            except Exception as e:
                logger.error(
                    f"Ошибка загрузки данных {symbol} для {exchange.id}: {str(e)}"
                )
                break

    def store_data(self, symbol, exchange, record):
        naive_date = datetime.datetime.utcfromtimestamp(record[0] / 1000)
        aware_date = timezone.make_aware(naive_date)

        try:
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
            logger.info(
                f"Запись успешно добавлена: {record} для {symbol} на {exchange.id}"
            )
        except Exception as e:
            logger.error(
                f"Ошибка добавления записи для {symbol} на {exchange.id}: {str(e)}"
            )

    def analyze_and_update(self):
        cryptocurrencies = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        models = self.get_models()
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

                df = self.prepare_data_for_analysis(data_all)
                if df is None:
                    logger.warning(
                        f"Не удалось подготовить данные для анализа {crypto}. Пропускаем."
                    )
                    continue

                logger.info(
                    f"Подготовка данных завершена для {crypto}, начинаем обучение моделей."
                )
                X, y = self.get_features_and_labels(df)

                scaler = StandardScaler()
                smote = SMOTE(random_state=42)

                logger.info(f"Масштабирование и ресэмплинг данных для {crypto}")
                X_scaled, y_resampled = self.scale_and_resample_data(
                    X, y, scaler, smote
                )

                logger.info(
                    f"Разделение данных на обучающую и тестовую выборки для {crypto}"
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled,
                    y_resampled,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_resampled,
                )

                total_predictions, correct_predictions = self.train_and_evaluate_models(
                    models,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    total_predictions,
                    correct_predictions,
                )

            except Exception as e:
                logger.error(f"Ошибка обработки {crypto}: {str(e)}")

        self.log_overall_accuracy(total_predictions, correct_predictions)

    def get_models(self):
        logger.info("Инициализация моделей для обучения.")
        models = {
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            ),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        logger.info("Модели успешно инициализированы.")
        return models

    def prepare_data_for_analysis(self, data_all):
        logger.info(
            f"Подготовка данных для анализа, количество записей: {len(data_all)}"
        )
        df = pd.DataFrame(list(data_all.values()))
        if len(df) < 14:
            logger.warning(f"Недостаточно данных для анализа. Пропускаем.")
            return None

        logger.info(f"Количество записей в DataFrame до обработки: {len(df)}")
        self.process_data(df)
        logger.info(
            f"Подготовка данных завершена, количество записей после обработки: {len(df)}"
        )
        return df

    def process_data(self, df):
        logger.info(f"Начало обработки данных: {len(df)} записей")

        close_prices = df["close_price"].tolist()

        df["SMA_14"] = (
            sma(close_prices, 14) if len(close_prices) >= 14 else [None] * len(df)
        )
        df["RSI_14"] = (
            rsi(close_prices, 14) if len(close_prices) >= 14 else [None] * len(df)
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

        logger.info(f"Количество записей после обработки: {len(df)}")
        logger.info(f"NaN в столбцах после обработки:\n{df.isnull().sum()}")

    def get_features_and_labels(self, df):
        logger.info(f"Получение признаков и меток из данных: {len(df)} записей")

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

        logger.info(f"Количество признаков: {X.shape[1]}, Количество меток: {len(y)}")

        return X, y

    def scale_and_resample_data(self, X, y, scaler, smote):
        logger.info(f"Начало обработки данных для масштабирования и ресемплинга")

        X = X.select_dtypes(include=["number"]).copy()
        logger.info(f"Количество числовых признаков: {X.shape[1]}")

        X.interpolate(method="linear", limit_direction="both", inplace=True)
        logger.info("Интерполяция пропусков данных выполнена")

        for col in X.columns:
            if X[col].isnull().sum() > 0:
                logger.warning(
                    f"Столбец {col} содержит пропущенные значения. Заполнение пропусков..."
                )
                X[col].fillna(method="bfill", inplace=True)
                X[col].fillna(method="ffill", inplace=True)

        logger.info("Ресемплинг с использованием SMOTE")
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Размер данных после ресемплинга: {X_resampled.shape[0]} примеров")

        logger.info("Масштабирование данных")
        X_scaled = scaler.fit_transform(X_resampled)
        logger.info("Масштабирование завершено")

        return X_scaled, y_resampled

    def save_predictions(self, predictions, probabilities, crypto, dates):
        logger.info(f"Сохранение предсказаний для криптовалюты: {crypto}")

        if isinstance(dates, pd.Index):
            dates = dates.to_list()
            logger.info(f"Конвертация индекса дат в список")

        for i, prediction in enumerate(predictions):
            logger.info(f"Сохранение предсказания для даты: {dates[i]}")

            MarketData.objects.update_or_create(
                cryptocurrency=crypto,
                date=dates[i],
                defaults={
                    "predicted_price_change": 1 if prediction else -1,
                    "probability_increase": probabilities[i],
                    "probability_decrease": 1 - probabilities[i],
                },
            )

            logger.info(
                f"Предсказание для {crypto} на {dates[i]} сохранено: {prediction}, вероятности: {probabilities[i]}, {1 - probabilities[i]}"
            )

    def train_and_evaluate_models(
        self,
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        total_predictions,
        correct_predictions,
    ):
        logger.info(f"Начало обучения и оценки моделей")
        tscv = TimeSeriesSplit(n_splits=5)

        for model_name, model in models.items():
            logger.info(f"Обработка модели: {model_name}")

            param_grid = {
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
                "Random Forest": {"n_estimators": [100, 150, 200]},
                "Logistic Regression": {"C": [0.1, 1, 10]},
            }.get(model_name, {})

            if param_grid:
                logger.info(
                    f"Параметры модели {model_name} для GridSearch: {param_grid}"
                )
                model = GridSearchCV(
                    model, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1
                )

            logger.info(f"Обучение модели {model_name}")
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            crypto = "BTC/USDT"
            dates = X_test.index
            self.save_predictions(predictions, probabilities, crypto, dates)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            logger.info(
                f"Точность модели {model_name}: {accuracy:.2%}, F1-score: {f1:.2%}"
            )

            total_predictions += len(predictions)
            correct_predictions += sum(predictions == y_test)

        logger.info(f"Завершено обучение и оценка моделей")
        return total_predictions, correct_predictions

    def log_overall_accuracy(self, total_predictions, correct_predictions):
        logger.info(f"Начало расчёта общей точности")

        overall_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        logger.info(f"Общая точность моделей: {overall_accuracy:.2%}")
        logger.info(
            f"Всего предсказаний: {total_predictions}, Правильных предсказаний: {correct_predictions}"
        )
