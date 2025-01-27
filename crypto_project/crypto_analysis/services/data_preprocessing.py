import logging
import pandas as pd
import time
from dotenv import load_dotenv
from django.utils import timezone
from django.db.models import Max
from crypto_analysis.models import IndicatorData, MarketData

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_for_all_cryptos():
    start_time = time.time()
    try:
        start_date_limit = timezone.now() - timezone.timedelta(days=1000)

        cryptos = (
            MarketData.objects.filter(date__gte=start_date_limit)
            .values_list("cryptocurrency", flat=True)
            .distinct()
        )
        cryptos_list = list(cryptos)

        all_data = {}
        for crypto in cryptos_list:
            logger.info(f"Запрос данных для {crypto} из базы данных.")
            if not crypto:
                logger.warning("Пустое имя криптовалюты. Пропускаем.")
                continue

            last_processed = IndicatorData.objects.filter(
                cryptocurrency=crypto
            ).aggregate(last_date=Max("date"))["last_date"]

            start_date = (
                last_processed
                if last_processed
                else timezone.now() - timezone.timedelta(days=730)
            )

            data_all = (
                MarketData.objects.filter(cryptocurrency=crypto, date__gt=start_date)
                .order_by("date")
                .values_list(
                    "date", "close_price", "high_price", "low_price", "cryptocurrency"
                )
            )

            logger.info(
                f"Получено {len(data_all)} записей для {crypto}. Пример: {data_all[:1]}."
            )
            if len(data_all) == 0:
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

            df = df.resample("h").last()
            df = df[~df.index.duplicated(keep="last")]

            all_data[crypto] = df

        logger.info(f"Данные получены за {time.time() - start_time:.2f} сек.")
        return all_data
    except Exception as e:
        logger.error(
            f"Ошибка при получении данных: {e} (затрачено {time.time() - start_time:.2f} сек.)"
        )
        return {}


def calculate_indicators(df: pd.DataFrame, crypto: str) -> pd.DataFrame:
    start_time = time.time()
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

        indicator_data_objects = []

        for indicator, periods in indicators.items():
            for period in periods if periods else [None]:
                if period and len(df) < period:
                    logger.warning(
                        f"Недостаточно данных для расчета индикатора {indicator} за {period} дней для {crypto}."
                    )
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

        for index, row in df.iterrows():
            for col in df.columns:
                if col not in [
                    "cryptocurrency",
                    "high_price",
                    "low_price",
                    "close_price",
                ]:
                    indicator_data_objects.append(
                        IndicatorData(
                            cryptocurrency=crypto,
                            date=index,
                            indicator_name=col,
                            value=row[col],
                        )
                    )

        if indicator_data_objects:
            IndicatorData.objects.bulk_create(indicator_data_objects, batch_size=1000)
            logger.info(
                f"Все индикаторы для криптовалюты {crypto} успешно рассчитаны и сохранены."
            )
        else:
            logger.warning(f"Нет данных для сохранения индикаторов для {crypto}.")

        logger.info(
            f"Индикаторы для {crypto} рассчитаны за {time.time() - start_time:.2f} сек."
        )
        return df
    except Exception as e:
        logger.error(
            f"Ошибка расчета индикаторов: {e} (затрачено {time.time() - start_time:.2f} сек.)"
        )
        return df


def calculate_and_store_correlations(all_data):
    start_time = time.time()
    try:
        if "BTC/USDT" not in all_data or "ETH/USDT" not in all_data:
            raise ValueError("Отсутствуют данные для BTC/USDT или ETH/USDT")

        last_correlation = IndicatorData.objects.filter(
            indicator_name__in=["BTC_Correlation", "ETH_Correlation"]
        ).aggregate(last_date=Max("date"))["last_date"]

        end_date = timezone.now()
        start_date = (
            last_correlation
            if last_correlation
            else end_date - timezone.timedelta(days=730)
        )

        for crypto, df in all_data.items():
            all_data[crypto] = df.loc[start_date:end_date]

        correlation_data_objects = []

        for crypto, df in all_data.items():
            df.loc[:, "returns"] = df["close_price"].pct_change(fill_method=None)

        returns_df = pd.concat(
            [df["returns"].rename(crypto) for crypto, df in all_data.items()], axis=1
        )

        btc_correlation = returns_df.corrwith(returns_df["BTC/USDT"])
        eth_correlation = returns_df.corrwith(returns_df["ETH/USDT"])

        for date, row in returns_df.iterrows():
            for crypto, correlation in btc_correlation.items():
                if pd.notna(correlation):
                    correlation_data_objects.append(
                        IndicatorData(
                            cryptocurrency=crypto,
                            date=date,
                            indicator_name="BTC_Correlation",
                            value=correlation,
                        )
                    )

            for crypto, correlation in eth_correlation.items():
                if pd.notna(correlation):
                    correlation_data_objects.append(
                        IndicatorData(
                            cryptocurrency=crypto,
                            date=date,
                            indicator_name="ETH_Correlation",
                            value=correlation,
                        )
                    )

        if correlation_data_objects:
            IndicatorData.objects.bulk_create(correlation_data_objects, batch_size=1000)
            logger.info("Корреляции успешно рассчитаны и сохранены для каждого часа.")
        else:
            logger.warning("Нет данных для сохранения корреляций.")

        logger.info(f"Корреляции рассчитаны за {time.time() - start_time:.2f} сек.")
    except Exception as e:
        logger.error(
            f"Ошибка при расчёте и сохранении корреляций: {e} (затрачено {time.time() - start_time:.2f} сек.)"
        )


def process_all_indicators():
    start_time = time.time()
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

        logger.info("Запуск расчёта корреляций для всех криптовалют.")
        try:
            calculate_and_store_correlations(cryptos_list)
            logger.info("Все корреляции успешно рассчитаны и сохранены.")
        except Exception as correlation_error:
            logger.error(f"Ошибка при расчёте корреляций: {correlation_error}")

    except Exception as e:
        logger.error(f"Ошибка при обработке всех индикаторов: {e}")
    logger.info(
        f"Обработка всех индикаторов завершена за {time.time() - start_time:.2f} сек."
    )


if __name__ == "__main__":
    logger.info("Начало обработки всех криптовалют.")
    process_all_indicators()
