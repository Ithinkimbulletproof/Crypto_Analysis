import logging
import pandas as pd
import time
import asyncio
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from dotenv import load_dotenv
from django.utils import timezone
from django.db.models import Max
from crypto_analysis.models import IndicatorData, MarketData
from asgiref.sync import sync_to_async

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async_get_cryptos = sync_to_async(
    lambda: list(
        MarketData.objects.filter(
            date__gte=timezone.now() - timezone.timedelta(days=1000)
        )
        .values_list("cryptocurrency", flat=True)
        .distinct()
    )
)

async_get_last_processed = sync_to_async(
    lambda crypto: IndicatorData.objects.filter(cryptocurrency=crypto).aggregate(
        last_date=Max("date")
    )["last_date"]
)

async_get_market_data = sync_to_async(
    lambda crypto, start_date: list(
        MarketData.objects.filter(cryptocurrency=crypto, date__gt=start_date)
        .order_by("date")
        .values_list("date", "close_price", "high_price", "low_price", "cryptocurrency")
    )
)

async_bulk_create = sync_to_async(IndicatorData.objects.bulk_create)


async def get_data_for_crypto(crypto: str) -> tuple[str, pd.DataFrame] | None:
    try:
        logger.info(f"Запрос данных для {crypto} из базы данных.")
        if not crypto:
            logger.warning("Пустое имя криптовалюты. Пропускаем.")
            return None

        last_processed = await async_get_last_processed(crypto)
        start_date = (
            last_processed
            if last_processed
            else timezone.now() - timezone.timedelta(days=730)
        )

        data_all = await async_get_market_data(crypto, start_date)

        logger.info(f"Получено {len(data_all)} записей для {crypto}.")
        if not data_all:
            logger.warning(f"Нет данных для {crypto}. Пропускаю.")
            return None

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

        df = df.resample("h").last()
        df = df[~df.index.duplicated(keep="last")]

        return (crypto, df)
    except Exception as e:
        logger.error(f"Ошибка при получении данных для {crypto}: {e}")
        return None


async def get_data_for_all_cryptos() -> dict:
    start_time = time.time()
    try:
        cryptos = await async_get_cryptos()
        tasks = [get_data_for_crypto(crypto) for crypto in cryptos]
        results = await asyncio.gather(*tasks)

        all_data = {}
        for result in results:
            if result:
                crypto, df = result
                all_data[crypto] = df

        logger.info(f"Данные получены за {time.time() - start_time:.2f} сек.")
        return all_data
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return {}


def calculate_indicators_sync(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
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
            for period in periods if periods else [None]:
                if period and len(df) < period:
                    continue

                if indicator == "price_change":
                    df[f"price_change_{period}d"] = (
                        df["close_price"].pct_change(periods=period, fill_method=None)
                        * 100
                    )
                elif indicator == "sma":
                    df[f"SMA_{period}"] = (
                        df["close_price"].rolling(window=period).mean()
                    )
                elif indicator == "volatility":
                    df[f"volatility_{period}d"] = (
                        df["close_price"].rolling(window=period).std()
                    )
                elif indicator == "rsi":
                    delta = df["close_price"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / (loss.replace(0, 1e-10))
                    df[f"RSI_{period}d"] = 100 - (100 / (1 + rs))
                elif indicator == "cci":
                    tp = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
                    sma_tp = tp.rolling(window=period).mean()
                    mean_deviation = tp.rolling(window=period).apply(
                        lambda x: (x - x.mean()).abs().mean(), raw=False
                    )
                    df[f"CCI_{period}d"] = (tp - sma_tp) / (0.015 * mean_deviation)
                elif indicator == "atr":
                    high_low = df["high_price"] - df["low_price"]
                    high_close = (df["high_price"] - df["close_price"].shift()).abs()
                    low_close = (df["low_price"] - df["close_price"].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(
                        axis=1
                    )
                    df[f"ATR_{period}"] = tr.rolling(window=period).mean()
                elif indicator == "bollinger_bands":
                    sma = df["close_price"].rolling(window=period).mean()
                    std = df["close_price"].rolling(window=period).std()
                    upper_band = sma + (std * 2)
                    lower_band = sma - (std * 2)
                    df[f"BB_upper_{period}"] = upper_band
                    df[f"BB_lower_{period}"] = lower_band
                elif indicator == "macd":
                    ema_12 = df["close_price"].ewm(span=12, adjust=False).mean()
                    ema_26 = df["close_price"].ewm(span=26, adjust=False).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    df["MACD_12_26"] = macd_line
                    df["MACD_signal_9"] = signal_line
                elif indicator == "stochastic_oscillator":
                    low_min = df["low_price"].rolling(window=period).min()
                    high_max = df["high_price"].rolling(window=period).max()
                    df[f"Stochastic_{period}"] = (
                        100 * (df["close_price"] - low_min) / (high_max - low_min)
                    )
                elif indicator == "lag_macd":
                    df[f"Lag_{period}"] = df["close_price"].shift(periods=period)

        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day in weekdays:
            df[f"seasonality_weekday_{day}"] = (df.index.day_name() == day).astype(int)

        for month in range(1, 13):
            df[f"seasonality_month_{month}"] = (df.index.month == month).astype(int)

        return df
    except Exception as e:
        logger.error(f"Ошибка расчета индикаторов: {e}")
        return df


async def calculate_indicators_for_crypto(crypto: str, df: pd.DataFrame) -> None:
    start_time = time.time()
    try:
        df = await asyncio.to_thread(calculate_indicators_sync, df.copy(), crypto)

        indicator_data_objects = [
            IndicatorData(
                cryptocurrency=crypto,
                date=index,
                indicator_name=col,
                value=row[col],
            )
            for index, row in df.iterrows()
            for col in df.columns
            if col not in ["cryptocurrency", "high_price", "low_price", "close_price"]
        ]

        if indicator_data_objects:
            await async_bulk_create(indicator_data_objects, batch_size=1000)
            logger.info(f"Индикаторы для {crypto} сохранены")
        else:
            logger.warning(f"Нет данных для сохранения ({crypto})")

        logger.info(
            f"Индикаторы для {crypto} обработаны за {time.time() - start_time:.2f} сек."
        )
    except Exception as e:
        logger.error(f"Ошибка обработки {crypto}: {e}")


async def calculate_and_store_correlations(all_data: dict) -> None:
    start_time = time.time()
    try:
        if "BTC/USDT" not in all_data or "ETH/USDT" not in all_data:
            raise ValueError("Отсутствуют данные для BTC/USDT или ETH/USDT")

        last_correlation = await sync_to_async(
            lambda: IndicatorData.objects.filter(
                indicator_name__in=["BTC_Correlation", "ETH_Correlation"]
            ).aggregate(last_date=Max("date"))["last_date"]
        )()

        end_date = timezone.now()
        start_date = (
            last_correlation
            if last_correlation
            else end_date - timezone.timedelta(days=730)
        )

        for crypto, df in all_data.items():
            all_data[crypto] = df.loc[start_date:end_date]

        correlation_data_objects = []
        returns_data = {}

        for crypto, df in all_data.items():
            df = df.copy()
            df["returns"] = df["close_price"].pct_change(fill_method=None)
            returns_data[crypto] = df["returns"]

        returns_df = pd.DataFrame(returns_data)

        btc_corr = returns_df.corrwith(returns_df["BTC/USDT"])
        eth_corr = returns_df.corrwith(returns_df["ETH/USDT"])

        for date in returns_df.index:
            for crypto, value in btc_corr.items():
                if pd.notna(value):
                    correlation_data_objects.append(
                        IndicatorData(
                            cryptocurrency=crypto,
                            date=date,
                            indicator_name="BTC_Correlation",
                            value=value,
                        )
                    )

            for crypto, value in eth_corr.items():
                if pd.notna(value):
                    correlation_data_objects.append(
                        IndicatorData(
                            cryptocurrency=crypto,
                            date=date,
                            indicator_name="ETH_Correlation",
                            value=value,
                        )
                    )

        if correlation_data_objects:
            await async_bulk_create(correlation_data_objects, batch_size=1000)
            logger.info("Корреляции сохранены")
        else:
            logger.warning("Нет данных для корреляций")

        logger.info(f"Корреляции обработаны за {time.time() - start_time:.2f} сек.")
    except Exception as e:
        logger.error(f"Ошибка расчета корреляций: {e}")


async def process_all_indicators() -> None:
    start_time = time.time()
    logger.info("Начало обработки индикаторов")
    try:
        all_data = await get_data_for_all_cryptos()
        if not all_data:
            logger.warning("Нет данных для обработки")
            return

        tasks = [
            calculate_indicators_for_crypto(crypto, df)
            for crypto, df in all_data.items()
        ]
        await asyncio.gather(*tasks)

        await calculate_and_store_correlations(all_data)

        logger.info(
            f"Полная обработка завершена за {time.time() - start_time:.2f} сек."
        )
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(process_all_indicators())
