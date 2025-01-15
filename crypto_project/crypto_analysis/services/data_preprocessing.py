import os
import logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from crypto_analysis.models import MarketData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def fetch_data_from_database(crypto: str) -> list:
    try:
        if not crypto:
            logger.warning("Пустое имя криптовалюты. Пропускаем.")
            return []

        data_all = (
            MarketData.objects.filter(cryptocurrency=crypto)
            .order_by("date")
            .values_list("date", "close", "cryptocurrency")
        )

        if not data_all:
            logger.warning(f"Нет данных для {crypto}. Пропускаем.")
            return []

        return list(data_all)
    except Exception as e:
        logger.error(f"Ошибка при запросе данных для {crypto}: {str(e)}")
        return []


def preprocess_data(data_all: list, volatility_window: int = 30) -> pd.DataFrame:
    if not data_all:
        logger.warning("Нет данных для анализа.")
        return pd.DataFrame()

    df = pd.DataFrame(data_all)
    df["close"].fillna(method="ffill", inplace=True)

    df[f"volatility_{volatility_window}"] = (
        df["close"].rolling(window=volatility_window).std()
    )

    return df


def calculate_market_volatility(
    all_cryptos_data: list, volatility_window: int = 30
) -> pd.DataFrame:
    df_all = pd.DataFrame(all_cryptos_data)
    df_all["close"].fillna(method="ffill", inplace=True)

    df_all[f"market_volatility_{volatility_window}"] = (
        df_all["close"].rolling(window=volatility_window).std()
    )

    return df_all


def split_data_by_period(df: pd.DataFrame, periods: list):
    now = datetime.now(timezone.utc)
    split_data = {}

    for period in periods:
        if period == 90:
            date_limit = now - timedelta(days=90)
        elif period == 180:
            date_limit = now - timedelta(days=180)
        elif period == 365:
            date_limit = now - timedelta(days=365)

        split_data[period] = df[df["date"] >= date_limit]

    return split_data


def save_to_csv(
    df: pd.DataFrame,
    cryptocurrency: str,
    period: str,
    file_path: str = "processed_data.csv",
):
    df["cryptocurrency"] = cryptocurrency
    df["period"] = period
    mode = "a" if os.path.exists(file_path) else "w"
    header = not os.path.exists(file_path)

    df.to_csv(file_path, mode=mode, header=header, index=False)
    logger.info(f"Данные сохранены для {cryptocurrency} ({period}) в {file_path}")


def save_data_to_csv_with_check(
    df: pd.DataFrame,
    cryptocurrency: str,
    period: str,
    file_path: str = "processed_data.csv",
):
    if os.path.exists(file_path):
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True

    df.to_csv(file_path, mode=mode, header=header, index=False)
    logger.info(f"Данные сохранены для {cryptocurrency} ({period}) в {file_path}")


def process_and_export_data(
    volatility_window: int = 30, periods: list = [90, 180, 365]
):
    cryptocurrencies = os.getenv("CRYPTOPAIRS").split(",")
    all_cryptos_data = []
    processed_cryptos_count = 0

    for crypto in cryptocurrencies:
        try:
            logger.info(f"Обработка данных для {crypto}")

            data_all = fetch_data_from_database(crypto)
            if not data_all:
                continue

            df = preprocess_data(data_all, volatility_window)
            if df.empty:
                continue

            all_cryptos_data.append(df[["date", "close", "cryptocurrency"]])

            split_data = split_data_by_period(df, periods)
            for period, data in split_data.items():
                save_data_to_csv_with_check(data, crypto, f"{period}_days")

            processed_cryptos_count += 1

        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")

    market_df = calculate_market_volatility(all_cryptos_data, volatility_window)
    save_data_to_csv_with_check(market_df, "market", "all_periods")

    log_overall_stats(processed_cryptos_count)


def log_overall_stats(processed_cryptos_count: int):
    logger.info("Обработка завершена.")
    logger.info(f"Обработано криптовалют: {processed_cryptos_count}")


if __name__ == "__main__":
    process_and_export_data(volatility_window=30)
