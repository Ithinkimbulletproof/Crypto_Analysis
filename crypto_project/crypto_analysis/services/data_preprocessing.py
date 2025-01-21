import os
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from django.db import IntegrityError
from crypto_analysis.models import IndicatorData, MarketData

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_indicators_to_db(df: pd.DataFrame, crypto: str):
    try:
        indicator_data_list = []
        for index, row in df.iterrows():
            for column in df.columns:
                if column != "cryptocurrency":
                    indicator_data = IndicatorData(
                        cryptocurrency=crypto,
                        date=index,
                        indicator_name=column,
                        value=row[column],
                    )

                    IndicatorData.objects.update_or_create(
                        cryptocurrency=indicator_data.cryptocurrency,
                        date=indicator_data.date,
                        indicator_name=indicator_data.indicator_name,
                        defaults={"value": indicator_data.value},
                    )

        logger.info(
            f"Успешно сохранено {len(indicator_data_list)} индикаторов для {crypto}."
        )
    except IntegrityError as e:
        logger.error(
            f"Ошибка при сохранении индикаторов в базу данных для {crypto}: {e}"
        )
    except Exception as e:
        logger.error(f"Неизвестная ошибка при сохранении данных для {crypto}: {e}")


def fetch_data_from_database(crypto: str) -> pd.DataFrame:
    logger.info(f"Запрос данных для {crypto} из базы данных.")
    try:
        if not crypto:
            logger.warning("Пустое имя криптовалюты. Пропускаем.")
            return pd.DataFrame()
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
            logger.warning(f"Нет данных для {crypto}. Пропускаем.")
            return pd.DataFrame()
        df = pd.DataFrame(
            data_all, columns=["date", "close", "high", "low", "cryptocurrency"]
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        logger.debug(f"Данные для {crypto} успешно преобразованы в DataFrame.")
        return df
    except Exception as e:
        logger.error(f"Ошибка при запросе данных для {crypto}: {e}")
        return pd.DataFrame()


def price_change(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [1, 7, 14, 30]

        for period in tqdm(periods, desc=f"Обработка {crypto}", unit="период"):
            df[f"price_change_{period}d"] = df["close"].pct_change(periods=period) * 100

        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете процентного изменения: {e}")
        return df


def sma(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [7, 14, 30, 50, 200]
        for period in periods:
            df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете скользящих средних: {e}")
        return df


def volatility(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [7, 14, 30, 60, 180]
        for period in periods:
            df[f"volatility_{period}d"] = df["close"].rolling(window=period).std()
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете волатильности: {e}")
        return df


def rsi(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [7, 14, 30, 90]
        for period in periods:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f"RSI_{period}d"] = 100 - (100 / (1 + rs))
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете RSI: {e}")
        return df


def cci(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [7, 14, 30]
        for period in periods:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = tp.rolling(window=period).mean()
            mean_deviation = tp.rolling(window=period).apply(
                lambda x: pd.Series(x).mad()
            )
            df[f"CCI_{period}d"] = (tp - sma_tp) / (0.015 * mean_deviation)
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете CCI: {e}")
        return df


def atr(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [14, 30, 60]
        for period in periods:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f"ATR_{period}"] = tr.rolling(window=period).mean()
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете ATR: {e}")
        return df


def bollinger_bands(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [14, 30, 50, 200]
        for period in periods:
            sma = df["close"].rolling(window=period).mean()
            std = df["close"].rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            df[f"BB_upper_{period}"] = upper_band
            df[f"BB_lower_{period}"] = lower_band
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете Bollinger Bands: {e}")
        return df


def macd(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26

        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        df["MACD_12_26"] = macd_line
        df["MACD_signal_9"] = signal_line
        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете MACD: {e}")
        return df


def stochastic_oscillator(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [7, 14, 30]

        for period in periods:
            low_min = df["low"].rolling(window=period).min()
            high_max = df["high"].rolling(window=period).max()
            df[f"Stochastic_{period}"] = (
                100 * (df["close"] - low_min) / (high_max - low_min)
            )

        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете стохастического осциллятора: {e}")
        return df


def lag_macd(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    try:
        periods = [12, 26, 9]

        for period in periods:
            df[f"Lag_{period}"] = df["close"].shift(periods=period)

        save_indicators_to_db(df, crypto)
        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете лагов: {e}")
        return df


def process_all_indicators(cryptos: list):
    logger.info("Начало обработки индикаторов для списка криптовалют.")
    try:
        for crypto in cryptos:
            logger.info(f"Запуск обработки данных для {crypto}")
            df = fetch_data_from_database(crypto)
            if df.empty:
                logger.warning(f"Данные для {crypto} не найдены. Пропускаю.")
                continue

            try:
                logger.info(f"Расчет изменения цены для {crypto}.")
                df = price_change(df, crypto)

                logger.info(f"Расчет SMA для {crypto}.")
                df = sma(df, crypto)

                logger.info(f"Расчет волатильности для {crypto}.")
                df = volatility(df, crypto)

                logger.info(f"Расчет RSI для {crypto}.")
                df = rsi(df, crypto)

                logger.info(f"Расчет CCI для {crypto}.")
                df = cci(df, crypto)

                logger.info(f"Расчет ATR для {crypto}.")
                df = atr(df, crypto)

                logger.info(f"Расчет полос Боллинджера для {crypto}.")
                df = bollinger_bands(df, crypto)

                logger.info(f"Расчет MACD для {crypto}.")
                df = macd(df, crypto)

                logger.info(f"Расчет стохастического осциллятора для {crypto}.")
                df = stochastic_oscillator(df, crypto)

                logger.info(f"Расчет MACD с лагом для {crypto}.")
                df = lag_macd(df, crypto)

                logger.info(f"Все индикаторы для {crypto} успешно рассчитаны.")
            except Exception as inner_e:
                logger.error(f"Ошибка при обработке индикаторов для {crypto}: {inner_e}")

    except Exception as e:
        logger.error(f"Ошибка при обработке всех индикаторов: {e}")
    logger.info("Обработка всех индикаторов завершена.")


if __name__ == "__main__":
    load_dotenv()
    cryptos_list = os.getenv("CRYPTOPAIRS", "").split(",")
    if cryptos_list:
        logger.info("Начало обработки для криптовалют: " + ", ".join(cryptos_list))
        process_all_indicators(cryptos_list)
    else:
        logger.warning("Список криптовалют пуст. Завершаю выполнение.")
