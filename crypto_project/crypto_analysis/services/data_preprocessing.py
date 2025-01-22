import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from django.utils import timezone
from django.db import IntegrityError
from crypto_analysis.models import IndicatorData, MarketData

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_indicators_to_db(df: pd.DataFrame, crypto: str):
    try:
        today = timezone.now()

        for index, row in df.iterrows():
            for column in df.columns:
                if column != "cryptocurrency":
                    indicator_data, created = IndicatorData.objects.update_or_create(
                        cryptocurrency=crypto,
                        date=today,
                        indicator_name=column,
                        defaults={"value": row[column]},
                    )
                    action = "обновлён" if not created else "создан"
                    logger.debug(
                        f"Индикатор {column} для {crypto} {action} за {today}."
                    )

        logger.info(
            f"Индикаторы для {crypto} успешно сохранены или обновлены за {today}."
        )
    except IntegrityError as e:
        logger.error(
            f"Ошибка при сохранении индикаторов в базу данных для {crypto}: {e}"
        )
    except Exception as e:
        logger.error(f"Неизвестная ошибка при сохранении данных для {crypto}: {e}")


def get_data_for_all_cryptos():
    try:
        cryptos = MarketData.objects.values_list("cryptocurrency", flat=True).distinct()
        cryptos_list = list(cryptos)

        all_data = {}
        for crypto in cryptos_list:
            logger.info(f"Запрос данных для {crypto} из базы данных.")
            if not crypto:
                logger.warning("Пустое имя криптовалюты. Пропускаем.")
                continue
            data_all = (
                MarketData.objects.filter(cryptocurrency=crypto)
                .order_by("date")
                .values_list(
                    "date", "close_price", "high_price", "low_price", "cryptocurrency"
                )
            )
            logger.info(
                f"Получено {len(data_all)} записей для {crypto}. Пример: {data_all[:1]}."
            )
            if not data_all:
                logger.warning(f"Нет данных для {crypto}. Пропускаю.")
                continue
            df = pd.DataFrame(
                data_all,
                columns=[
                    "date",
                    "close_price",
                    "high_price",
                    "low_price",
                    "cryptocurrency",
                ],
            )
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            logger.debug(f"Данные для {crypto} успешно преобразованы в DataFrame.")
            all_data[crypto] = df

        return all_data
    except Exception as e:
        logger.error(f"Ошибка при получении данных для криптовалют: {e}")
        return {}


def calculate_indicators(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        indicators = {
            "price_change": [1, 7, 14, 30],
            "sma": [7, 14, 30, 50, 200],
            "volatility": [7, 14, 30, 60, 180],
            "rsi": [7, 14, 30, 90],
            "cci": [7, 14, 30],
            "atr": [14, 30, 60],
            "bollinger_bands": [14, 30, 50, 200],
            "macd": None,
            "stochastic_oscillator": [7, 14, 30],
            "lag_macd": [12, 26, 9],
        }

        for indicator, periods in indicators.items():
            if indicator == "price_change":
                for period in periods:
                    df[f"price_change_{period}d"] = (
                        df["close_price"].pct_change(periods=period) * 100
                    )
                logger.info(
                    f"Расчёт индикатора price_change для периодов {periods} завершён."
                )
            elif indicator == "sma":
                for period in periods:
                    df[f"SMA_{period}"] = (
                        df["close_price"].rolling(window=period).mean()
                    )
                logger.info(f"Расчёт индикатора sma для периодов {periods} завершён.")
            elif indicator == "volatility":
                for period in periods:
                    df[f"volatility_{period}d"] = (
                        df["close_price"].rolling(window=period).std()
                    )
                logger.info(
                    f"Расчёт индикатора volatility для периодов {periods} завершён."
                )
            elif indicator == "rsi":
                for period in periods:
                    delta = df["close_price"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    df[f"RSI_{period}d"] = 100 - (100 / (1 + rs))
                logger.info(f"Расчёт индикатора rsi для периодов {periods} завершён.")
            elif indicator == "cci":
                for period in periods:
                    tp = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
                    sma_tp = tp.rolling(window=period).mean()
                    mean_deviation = tp.rolling(window=period).apply(
                        lambda x: (x - x.mean()).abs().mean(), raw=False
                    )
                    df[f"CCI_{period}d"] = (tp - sma_tp) / (0.015 * mean_deviation)
                logger.info(f"Расчёт индикатора cci для периодов {periods} завершён.")
            elif indicator == "atr":
                for period in periods:
                    high_low = df["high_price"] - df["low_price"]
                    high_close = (df["high_price"] - df["close_price"].shift()).abs()
                    low_close = (df["low_price"] - df["close_price"].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(
                        axis=1
                    )
                    df[f"ATR_{period}"] = tr.rolling(window=period).mean()
                logger.info(f"Расчёт индикатора atr для периодов {periods} завершён.")
            elif indicator == "bollinger_bands":
                for period in periods:
                    sma = df["close_price"].rolling(window=period).mean()
                    std = df["close_price"].rolling(window=period).std()
                    upper_band = sma + (std * 2)
                    lower_band = sma - (std * 2)
                    df[f"BB_upper_{period}"] = upper_band
                    df[f"BB_lower_{period}"] = lower_band
                logger.info(
                    f"Расчёт индикатора bollinger_bands для периодов {periods} завершён."
                )
            elif indicator == "macd":
                ema_12 = df["close_price"].ewm(span=12, adjust=False).mean()
                ema_26 = df["close_price"].ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                df["MACD_12_26"] = macd_line
                df["MACD_signal_9"] = signal_line
                logger.info(f"Расчёт индикатора macd завершён.")
            elif indicator == "stochastic_oscillator":
                for period in periods:
                    low_min = df["low_price"].rolling(window=period).min()
                    high_max = df["high_price"].rolling(window=period).max()
                    df[f"Stochastic_{period}"] = (
                        100 * (df["close_price"] - low_min) / (high_max - low_min)
                    )
                logger.info(
                    f"Расчёт индикатора stochastic_oscillator для периодов {periods} завершён."
                )
            elif indicator == "lag_macd":
                for period in periods:
                    df[f"Lag_{period}"] = df["close_price"].shift(periods=period)
                logger.info(
                    f"Расчёт индикатора lag_macd для периодов {periods} завершён."
                )

        save_indicators_to_db(df, crypto)
        logger.info(f"Данные для криптовалюты {crypto} сохранены в базу данных.")
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчёте индикаторов: {e}")
        return df


def process_all_indicators():
    logger.info("Начало обработки индикаторов для списка криптовалют.")
    try:
        cryptos_list = get_data_for_all_cryptos()
        if not cryptos_list:
            logger.warning("Нет криптовалют для обработки. Завершаю выполнение.")
            return

        for crypto, df in cryptos_list.items():
            if df.empty:
                logger.warning(f"Данные для {crypto} не найдены. Пропускаю.")
                continue

            logger.info(f"Запуск обработки данных для {crypto}.")
            try:
                df = calculate_indicators(df, crypto)

                logger.info(f"Все индикаторы для {crypto} успешно рассчитаны.")
            except Exception as inner_e:
                logger.error(
                    f"Ошибка при обработке индикаторов для {crypto}: {inner_e}"
                )

    except Exception as e:
        logger.error(f"Ошибка при обработке всех индикаторов: {e}")
    logger.info("Обработка всех индикаторов завершена.")


if __name__ == "__main__":
    logger.info("Начало обработки всех криптовалют.")
    process_all_indicators()
