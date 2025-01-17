import os
import logging
import pandas as pd
from dotenv import load_dotenv
from crypto_analysis.models import MarketData
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def fetch_data_from_database(crypto: str) -> list:
    logger.info(f"Запрос данных для {crypto} из базы данных.")
    try:
        if not crypto:
            logger.warning("Пустое имя криптовалюты. Пропускаем.")
            return []
        data_all = (
            MarketData.objects.filter(cryptocurrency=crypto)
            .order_by("date")
            .values_list(
                "date", "close_price", "high_price", "low_price", "cryptocurrency"
            )
        )
        logger.info(f"Получено {len(data_all)} записей для {crypto}. Пример записи: {data_all[:1]}")
        if not data_all:
            logger.warning(f"Нет данных для {crypto}. Пропускаем.")
            return []
        df = pd.DataFrame(data_all, columns=["date", "close", "high", "low", "cryptocurrency"])
        logger.info(f"DataFrame создан успешно. Пример первой записи: {df.head(1)}")
        logger.info(f"Количество строк в DataFrame: {len(df)}")
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        logger.error(f"Ошибка при запросе данных для {crypto}: {str(e)}")
        return []


def preprocess_data(
        data_all: list, volatility_window: int = 30, k_window: int = 14
) -> pd.DataFrame:
    if not data_all:
        logger.warning("Нет данных для анализа.")
        return pd.DataFrame()
    try:
        df = pd.DataFrame(
            data_all, columns=["date", "close", "high", "low", "cryptocurrency"]
        )
        logger.info(f"DataFrame создан успешно в preprocess_data. Пример первой записи: {df.head(1)}")
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Дата преобразована в datetime. Пример первой записи: {df.head(1)}")
        df.set_index("date", inplace=True)
        logger.info(f"Индекс установлен. Пример первой записи: {df.head(1)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("Индекс не является DatetimeIndex!")
            return pd.DataFrame()
        df["close"] = df["close"].interpolate(method="time")
        logger.info(f"Close цена интерполирована. Пример первой записи: {df.head(1)}")
        df["high"] = df["high"].interpolate(method="time")
        logger.info(f"High цена интерполирована. Пример первой записи: {df.head(1)}")
        df["low"] = df["low"].interpolate(method="time")
        logger.info(f"Low цена интерполирована. Пример первой записи: {df.head(1)}")
        df["close"] = df["close"].fillna(df["close"].mean())
        logger.info(f"Close цена заполнена средним значением. Пример первой записи: {df.head(1)}")
        df["high"] = df["high"].fillna(df["high"].mean())
        logger.info(f"High цена заполнена средним значением. Пример первой записи: {df.head(1)}")
        df["low"] = df["low"].fillna(df["low"].mean())
        logger.info(f"Low цена заполнена средним значением. Пример первой записи: {df.head(1)}")
        df["price_change_24h"] = df["close"].pct_change(periods=1) * 100
        logger.info(f"Price change 24h рассчитан. Пример первой записи: {df.head(1)}")
        df["price_change_7d"] = df["close"].pct_change(periods=7) * 100
        logger.info(f"Price change 7d рассчитан. Пример первой записи: {df.head(1)}")
        df["SMA_30"] = df["close"].rolling(window=30).mean()
        logger.info(f"SMA 30 рассчитан. Пример первой записи: {df.head(1)}")
        df["SMA_90"] = df["close"].rolling(window=90).mean()
        logger.info(f"SMA 90 рассчитан. Пример первой записи: {df.head(1)}")
        df["SMA_180"] = df["close"].rolling(window=180).mean()
        logger.info(f"SMA 180 рассчитан. Пример первой записи: {df.head(1)}")
        df["SMA_30"] = df["SMA_30"].interpolate(method="time")
        logger.info(f"SMA 30 интерполирован. Пример первой записи: {df.head(1)}")
        df["SMA_90"] = df["SMA_90"].interpolate(method="time")
        logger.info(f"SMA 90 интерполирован. Пример первой записи: {df.head(1)}")
        df["SMA_180"] = df["SMA_180"].interpolate(method="time")
        logger.info(f"SMA 180 интерполирован. Пример первой записи: {df.head(1)}")
        df[f"volatility_{volatility_window}"] = (
            df["close"].rolling(window=volatility_window).std()
        )
        logger.info(f"Волатильность рассчитана. Пример первой записи: {df.head(1)}")
        df[f"volatility_{volatility_window}"] = df[
            f"volatility_{volatility_window}"
        ].interpolate(method="time")
        logger.info(f"Волатильность интерполирована. Пример первой записи: {df.head(1)}")
        df["future_24h_change"] = df["close"].shift(-1) - df["close"]
        logger.info(f"Future 24h change рассчитан. Пример первой записи: {df.head(1)}")
        df["future_24h_up"] = (df["future_24h_change"] > 0).astype(int)
        logger.info(f"Future 24h up рассчитан. Пример первой записи: {df.head(1)}")
        df["lowest_low"] = df["low"].rolling(window=k_window).min()
        logger.info(f"Lowest low рассчитан. Пример первой записи: {df.head(1)}")
        df["highest_high"] = df["high"].rolling(window=k_window).max()
        logger.info(f"Highest high рассчитан. Пример первой записи: {df.head(1)}")
        df.reset_index(inplace=True)
        logger.info(f"Индекс сброшен. Пример первой записи: {df.head(1)}")
        return df
    except Exception as e:
        logger.error(f"Ошибка при предобработке данных: {str(e)}")
        return pd.DataFrame()

def split_data_by_period(df: pd.DataFrame, periods: list):
    logger.info(f"Разделение данных на периоды: {periods}")
    now = datetime.now(timezone.utc)
    split_data = {}
    for period in periods:
        date_limit = now - timedelta(days=period)
        split_data[period] = df[df["date"] >= date_limit]
    return split_data

def save_to_csv(
    df: pd.DataFrame,
    cryptocurrency: str,
    period: str,
    file_path: str = "processed_data.csv",
):
    try:
        logger.info(f"Сохранение данных для {cryptocurrency} ({period}) в {file_path}.")
        df = df.copy()
        df.loc[:, "cryptocurrency"] = cryptocurrency
        df.loc[:, "period"] = period
        mode = "a" if os.path.exists(file_path) else "w"
        header = not os.path.exists(file_path)
        df.to_csv(file_path, mode=mode, header=header, index=False)
        logger.info(f"Данные успешно сохранены для {cryptocurrency} ({period}) в {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных для {cryptocurrency}: {e}")
        raise

def process_and_export_data(
    volatility_window: int = 30, periods: list = [90, 180, 365]
):
    logger.info("Запуск процесса обработки и экспорта данных.")
    cryptocurrencies = os.getenv("CRYPTOPAIRS").split(",")
    all_cryptos_data = []
    processed_cryptos_count = 0
    for crypto in cryptocurrencies:
        try:
            logger.info(f"Обработка данных для {crypto}")
            data_all = fetch_data_from_database(crypto)
            if not data_all:
                logger.info(f"Нет данных для {crypto}, пропускаем.")
                continue
            df = preprocess_data(data_all, volatility_window)
            if df.empty:
                logger.info(f"Предобработанные данные для {crypto} пустые, пропускаем.")
                continue
            all_cryptos_data.append(
                df[["date", "close", "high", "low", "cryptocurrency"]]
            )
            split_data = split_data_by_period(df, periods)
            for period, data in split_data.items():
                save_to_csv(data, crypto, f"{period}_days")
            processed_cryptos_count += 1
        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")
    if all_cryptos_data:
        market_df = pd.concat(all_cryptos_data)
        save_to_csv(market_df, "market", "all_periods")
    log_overall_stats(processed_cryptos_count)

def log_overall_stats(processed_cryptos_count: int):
    logger.info("Обработка завершена.")
    logger.info(f"Обработано криптовалют: {processed_cryptos_count}")

if __name__ == "__main__":
    process_and_export_data(volatility_window=30)
