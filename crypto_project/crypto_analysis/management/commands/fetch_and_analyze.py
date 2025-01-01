import ccxt
import logging
import time
import datetime
import pandas as pd
from django.utils import timezone
from django.core.management.base import BaseCommand
from crypto_analysis.models import MarketData
from pyti.smoothed_moving_average import smoothed_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
        exchanges = [
            ccxt.binance(),
            ccxt.kraken(),
            ccxt.bitfinex(),
            ccxt.kucoin(),
            ccxt.gemini(),
            ccxt.huobi(),
            ccxt.okx(),
            ccxt.bybit(),
        ]

        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "LTC/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "DOGE/USDT",
            "MATIC/USDT",
            "DOT/USDT",
            "SHIB/USDT",
            "AVAX/USDT",
            "FTM/USDT",
            "LUNA/USDT",
            "TON/USDT",
        ]

        timeframe = "1h"
        since = ccxt.binance().parse8601("2021-01-01T00:00:00Z")

        logger.info(f"Получение данных для криптовалютных пар: {symbols}")

        for exchange in exchanges:
            try:
                logger.info(f"Загружаем рынки для биржи: {exchange.id}")
                exchange.load_markets()

                for symbol in symbols:
                    base, quote = symbol.split("/")
                    market_symbol = f"{base}/{quote}".upper()

                    try:
                        if market_symbol in exchange.markets:
                            logger.info(
                                f"Запрос данных для {market_symbol} на бирже {exchange.id}"
                            )

                            data = exchange.fetch_ohlcv(
                                market_symbol, timeframe, since=since, limit=1000
                            )

                            if isinstance(data, list):
                                logger.info(
                                    f"Получены данные для {market_symbol} на бирже {exchange.id}"
                                )
                                for record in data:
                                    if isinstance(record, list) and len(record) >= 6:
                                        timestamp = time.gmtime(record[0] / 1000)
                                        naive_date = datetime.datetime(*timestamp[:6])
                                        aware_date = timezone.make_aware(naive_date)

                                        if not MarketData.objects.filter(
                                            cryptocurrency=symbol, date=aware_date
                                        ).exists():
                                            logger.info(
                                                f"Сохраняем данные для {market_symbol} на {aware_date}"
                                            )
                                            MarketData.objects.create(
                                                cryptocurrency=symbol,
                                                date=aware_date,
                                                open_price=record[1],
                                                high_price=record[2],
                                                low_price=record[3],
                                                close_price=record[4],
                                                volume=record[5],
                                            )
                                    else:
                                        logger.warning(
                                            f"Неверная структура данных для пары {symbol} на бирже {exchange.id}"
                                        )
                            else:
                                logger.warning(
                                    f"Ожидался список данных, но получен другой тип для пары {symbol} на бирже {exchange.id}"
                                )
                        else:
                            logger.warning(
                                f"Пара {market_symbol} не найдена на бирже {exchange.id}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обработке пары {symbol} на бирже {exchange.id}: {str(e)}"
                        )

            except Exception as e:
                logger.error(f"Ошибка при работе с биржей {exchange.id}: {str(e)}")

        logger.info("Загрузка данных завершена")

    def analyze_and_update(self):
        cryptocurrencies = [
            "BTC/USDT",
            "ETH/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "LTC/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "DOGE/USDT",
            "MATIC/USDT",
            "DOT/USDT",
            "SHIB/USDT",
            "AVAX/USDT",
            "FTM/USDT",
            "LUNA/USDT",
            "TON/USDT",
        ]

        total_predictions = 0
        correct_predictions = 0

        for crypto in cryptocurrencies:
            data_all = MarketData.objects.filter(cryptocurrency=crypto).order_by(
                "-date"
            )
            last_90_days = timezone.now() - datetime.timedelta(days=90)
            data_recent = data_all.filter(date__gte=last_90_days)

            if not data_recent.exists():
                logger.warning(f"Нет данных для {crypto} за последние 90 дней.")
                continue

            df_all = pd.DataFrame(list(data_all.values())).copy()
            df_recent = pd.DataFrame(list(data_recent.values())).copy()

            for df in [df_all, df_recent]:
                df["SMA_14"] = sma(df["close_price"].tolist(), 14)
                df["RSI_14"] = rsi(df["close_price"].tolist(), 14)
                df["volatility"] = df["close_price"].rolling(window=14).std()
                df["volume_change"] = df["volume"].pct_change()
                df["price_change"] = df["close_price"].pct_change().shift(-24)
                df["actual_change"] = (df["price_change"] > 0).astype(int)
                df.dropna(inplace=True)

            try:
                X_all = df_all[
                    [
                        "close_price",
                        "volume",
                        "SMA_14",
                        "RSI_14",
                        "volatility",
                        "volume_change",
                    ]
                ]
                y_all = df_all["actual_change"]

                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                model.fit(X_all, y_all)

                X_recent = df_recent[
                    [
                        "close_price",
                        "volume",
                        "SMA_14",
                        "RSI_14",
                        "volatility",
                        "volume_change",
                    ]
                ]
                predictions = model.predict(X_recent)
                probabilities = model.predict_proba(X_recent)

                df_recent["predicted_price_change"] = predictions
                df_recent["probability_increase"] = probabilities[:, 1]
                df_recent["probability_decrease"] = probabilities[:, 0]

                total_predictions += len(predictions)
                correct_predictions += (predictions == df_recent["actual_change"]).sum()

                for index, row in df_recent.iterrows():
                    logger.info(
                        f"{crypto}: Дата {row['date']} | Рост: {row['probability_increase']:.2%} | "
                        f"Падение: {row['probability_decrease']:.2%} | "
                        f"Факт: {'Рост' if row['actual_change'] == 1 else 'Падение'}"
                    )

                update_objects = []
                for _, row in df_recent.iterrows():
                    obj = MarketData.objects.get(id=row["id"])
                    obj.sma_14 = row["SMA_14"]
                    obj.rsi_14 = row["RSI_14"]
                    obj.predicted_price_change = row["predicted_price_change"]
                    obj.probability_increase = row["probability_increase"]
                    obj.probability_decrease = row["probability_decrease"]
                    update_objects.append(obj)

                MarketData.objects.bulk_update(
                    update_objects,
                    [
                        "sma_14",
                        "rsi_14",
                        "predicted_price_change",
                        "probability_increase",
                        "probability_decrease",
                    ],
                )
                logger.info(
                    f"Обновлены данные для {crypto}: {len(update_objects)} записей."
                )

            except Exception as e:
                logger.error(f"Ошибка при анализе данных для {crypto}: {str(e)}")

        overall_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        logger.info(f"Общая точность модели: {overall_accuracy:.2%}")
