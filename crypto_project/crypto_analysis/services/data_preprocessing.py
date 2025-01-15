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
    df["date"] = pd.to_datetime(df["date"])
    df["close"].fillna(method="ffill", inplace=True)

    df["price_change_24h"] = df["close"].pct_change(periods=1) * 100
    df["price_change_7d"] = df["close"].pct_change(periods=7) * 100

    df["SMA_30"] = df["close"].rolling(window=30).mean()
    df["SMA_90"] = df["close"].rolling(window=90).mean()
    df["SMA_180"] = df["close"].rolling(window=180).mean()

    df[f"volatility_{volatility_window}"] = (
        df["close"].rolling(window=volatility_window).std()
    )

    df["future_24h_change"] = df["close"].shift(-1) - df["close"]
    df["future_24h_up"] = (df["future_24h_change"] > 0).astype(int)

    return df.dropna()


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
                save_to_csv(data, crypto, f"{period}_days")

            processed_cryptos_count += 1

        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")

    market_df = pd.concat(all_cryptos_data)
    save_to_csv(market_df, "market", "all_periods")

    log_overall_stats(processed_cryptos_count)


def log_overall_stats(processed_cryptos_count: int):
    logger.info("Обработка завершена.")
    logger.info(f"Обработано криптовалют: {processed_cryptos_count}")


if __name__ == "__main__":
    process_and_export_data(volatility_window=30)
