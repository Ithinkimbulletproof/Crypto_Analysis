import ta
import logging
import pandas as pd
from django.db import transaction
from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from crypto_analysis.models import MarketData, IndicatorData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data():
    start_date = make_aware(datetime.now() - timedelta(days=2000))
    market_data = MarketData.objects.filter(date__gte=start_date).order_by(
        "cryptocurrency", "date"
    )

    if not market_data.exists():
        logger.info("Данных в MarketData не найдено.")
        return

    df = pd.DataFrame.from_records(
        market_data.values(
            "cryptocurrency",
            "date",
            "close_price",
            "volume",
            "open_price",
            "high_price",
            "low_price",
        )
    )
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Загружено {len(df)} записей из MarketData.")

    calc_start_date = make_aware(datetime.now() - timedelta(days=1900))

    grouped = df.groupby("cryptocurrency")

    existing_keys = set(
        IndicatorData.objects.filter(date__gte=start_date).values_list(
            "cryptocurrency", "date", "indicator_name"
        )
    )

    new_entries = []
    for crypto, data in grouped:
        logger.info(f"Обрабатываем {crypto}: всего записей = {len(data)}")
        data_calc = data[data["date"] >= calc_start_date]
        logger.info(
            f"Для {crypto} используется {len(data_calc)} записей за последние 1900 дней для расчёта."
        )

        if data_calc.empty:
            logger.info(f"Для {crypto} нет записей за последние 1900 дней. Пропускаем.")
            continue

        indicators = calculate_indicators(data_calc)
        indicators = indicators.dropna(subset=["value"])
        logger.info(
            f"Для {crypto} получено {len(indicators)} валидных записей индикаторов."
        )

        for _, row in indicators.iterrows():
            key = (crypto, row["date"], row["indicator"])
            if key not in existing_keys:
                new_entries.append(
                    IndicatorData(
                        cryptocurrency=crypto,
                        date=row["date"],
                        indicator_name=row["indicator"],
                        value=row["value"],
                    )
                )

    logger.info(f"Всего новых записей для вставки: {len(new_entries)}")
    if new_entries:
        with transaction.atomic():
            IndicatorData.objects.bulk_create(new_entries, batch_size=1000)
            logger.info("Новые записи успешно сохранены в БД.")


def calculate_indicators(data):
    indicators = pd.DataFrame()

    logger.info("Вычисляем RSI 14...")
    indicators["RSI 14"] = ta.momentum.RSIIndicator(
        data["close_price"], window=14
    ).rsi()

    logger.info("Вычисляем MACD...")
    macd_indicator = ta.trend.MACD(data["close_price"])
    indicators["MACD"] = macd_indicator.macd()
    indicators["MACD Signal"] = macd_indicator.macd_signal()
    indicators["MACD Hist"] = macd_indicator.macd_diff()

    logger.info("Вычисляем SMA 20...")
    indicators["SMA 20"] = ta.trend.SMAIndicator(
        data["close_price"], window=20
    ).sma_indicator()

    logger.info("Вычисляем EMA 50...")
    indicators["EMA 50"] = ta.trend.EMAIndicator(
        data["close_price"], window=50
    ).ema_indicator()

    logger.info("Вычисляем Bollinger Bands...")
    bb_indicator = ta.volatility.BollingerBands(data["close_price"], window=20)
    indicators["BBANDS Upper"] = bb_indicator.bollinger_hband()
    indicators["BBANDS Middle"] = bb_indicator.bollinger_mavg()
    indicators["BBANDS Lower"] = bb_indicator.bollinger_lband()

    logger.info("Вычисляем ATR 14...")
    indicators["ATR 14"] = ta.volatility.AverageTrueRange(
        data["high_price"], data["low_price"], data["close_price"], window=14
    ).average_true_range()

    logger.info("Вычисляем Stochastic Oscillator...")
    stoch_indicator = ta.momentum.StochasticOscillator(
        data["high_price"], data["low_price"], data["close_price"], window=14
    )
    indicators["STOCH SlowK"] = stoch_indicator.stoch()
    indicators["STOCH SlowD"] = stoch_indicator.stoch_signal()

    logger.info("Вычисляем VWAP...")
    indicators["VWAP"] = calculate_vwap(data)

    logger.info("Вычисляем ADX 14...")
    indicators["ADX 14"] = ta.trend.ADXIndicator(
        data["high_price"], data["low_price"], data["close_price"], window=14
    ).adx()

    logger.info("Вычисляем OBV...")
    indicators["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        data["close_price"], data["volume"]
    ).on_balance_volume()

    indicators["date"] = data["date"]
    indicators = indicators.melt(
        id_vars=["date"], var_name="indicator", value_name="value"
    )

    logger.info("Все индикаторы вычислены.")
    return indicators


def calculate_vwap(data):
    return (data["close_price"] * data["volume"]).cumsum() / data["volume"].cumsum()
