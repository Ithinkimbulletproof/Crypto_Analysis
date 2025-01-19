import os
import logging
import pandas as pd
from dotenv import load_dotenv
from crypto_analysis.models import MarketData, PreprocessedData
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
        logger.info(
            f"Получено {len(data_all)} записей для {crypto}. Пример записи: {data_all[:1]}"
        )
        if not data_all:
            logger.warning(f"Нет данных для {crypto}. Пропускаем.")
            return []
        df = pd.DataFrame(
            data_all, columns=["date", "close", "high", "low", "cryptocurrency"]
        )
        logger.info(f"DataFrame создан успешно. Пример первой записи: {df.head(1)}")
        logger.info(f"Количество строк в DataFrame: {len(df)}")
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        logger.error(f"Ошибка при запросе данных для {crypto}: {str(e)}")
        return []


def ensure_index(df: pd.DataFrame, index_col: str):
    if index_col not in df.columns:
        df.reset_index(inplace=True)


def preprocess_data(
    data_all: list, volatility_window: int = 30, k_window: int = 14
) -> pd.DataFrame:
    if not data_all:
        logger.warning("Список данных пуст. Возвращается пустой DataFrame.")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(
            data_all, columns=["date", "close", "high", "low", "cryptocurrency"]
        )
        logger.info(f"DataFrame создан успешно. Пример первой записи: {df.head(1)}")

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        for col in ["close", "high", "low"]:
            df[col] = df[col].interpolate(method="time").bfill().ffill()
            logger.info(
                f"{col.capitalize()} цена обработана. Пример первой записи: {df.head(1)}"
            )

        df["price_change_24h"] = df["close"].pct_change(periods=24)
        df["price_change_7d"] = df["close"].pct_change(periods=7 * 24)

        rolling_windows = [30, 90, 180]
        for window in rolling_windows:
            df[f"SMA_{window}"] = df["close"].rolling(window=window).mean()
            df[f"volatility_{window}"] = df["close"].rolling(window=window).std()
            logger.info(f"SMA и волатильность для {window} дней рассчитаны.")

        df.dropna(inplace=True)
        logger.info(
            f"Пустые строки удалены. Пример первой записи после очистки: {df.head(1)}"
        )

        return df.reset_index()
    except Exception as e:
        logger.error(f"Ошибка при предобработке данных: {e}", exc_info=True)
        return pd.DataFrame()


def split_data_by_period(df: pd.DataFrame, periods: list) -> dict:
    logger.info(f"Разделение данных на периоды: {periods}")
    try:
        ensure_index(df, "date")
        now = datetime.now(timezone.utc)
        split_data = {}
        for period in periods:
            date_limit = now - timedelta(days=period)
            filtered_df = df[df["date"] >= date_limit].copy()
            split_data[period] = filtered_df
            logger.info(f"Данные для {period} дней: {len(filtered_df)} записей.")
        return split_data
    except Exception as e:
        logger.error(f"Ошибка при разделении данных на периоды: {e}", exc_info=True)
        return {}


def save_to_database(df: pd.DataFrame, cryptocurrency: str, period: int):
    try:
        logger.info(f"Сохранение данных для {cryptocurrency} ({period} дней) в базу данных.")

        df = df.copy()
        df["cryptocurrency"] = cryptocurrency
        df["period"] = period

        preprocessed_data_list = [
            PreprocessedData(
                date=row["date"],
                close_price=row["close"],
                high_price=row["high"],
                low_price=row["low"],
                cryptocurrency=row["cryptocurrency"],
                period=row["period"],
                price_change_24h=row.get("price_change_24h", None),
                SMA_30=row.get("SMA_30", None),
                volatility_30=row.get("volatility_30", None),
                SMA_90=row.get("SMA_90", None),
                volatility_90=row.get("volatility_90", None),
                SMA_180=row.get("SMA_180", None),
                volatility_180=row.get("volatility_180", None)
            )
            for _, row in df.iterrows()
        ]

        PreprocessedData.objects.bulk_create(preprocessed_data_list)

        logger.info(f"Данные успешно сохранены для {cryptocurrency} ({period} дней) в базу данных.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных для {cryptocurrency}: {e}", exc_info=True)
        raise


def process_and_export_data(
    volatility_window: int = 30, periods: list = [30, 90, 180, 365]
):
    logger.info("Запуск процесса обработки и экспорта данных.")
    cryptocurrencies = os.getenv("CRYPTOPAIRS", "").split(",")
    if not cryptocurrencies:
        logger.error("Переменная CRYPTOPAIRS не установлена или пуста.")
        return

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

            split_data = split_data_by_period(df, periods)
            for period, data in split_data.items():
                save_to_database(data, crypto, period)

            all_cryptos_data.append(
                df[["date", "close", "high", "low", "cryptocurrency"]]
            )
            processed_cryptos_count += 1
        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")

    if all_cryptos_data:
        market_df = pd.concat(all_cryptos_data, ignore_index=True)
        save_to_database(market_df, "market", "all_periods")

    log_overall_stats(processed_cryptos_count)


def log_overall_stats(processed_cryptos_count: int):
    logger.info("Обработка завершена.")
    logger.info(f"Обработано криптовалют: {processed_cryptos_count}")


if __name__ == "__main__":
    process_and_export_data(volatility_window=30)
