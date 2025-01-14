import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_volatility(df, windows=[30, 90, 180]):
    try:
        if df is None or "close_price" not in df.columns:
            raise ValueError("Столбец 'close_price' отсутствует в DataFrame.")

        df["close_price"] = df["close_price"].ffill()

        if not isinstance(windows, list):
            windows = [windows]

        for window in windows:
            col_name = f"volatility_{window}_days"
            df[col_name] = df["close_price"].rolling(window=window).std()

        logger.info(f"Волатильность рассчитана для окон: {windows}")

        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете волатильности: {str(e)}")
        return None


def adjust_threshold_based_on_volatility(volatility, default_threshold=0.5):
    try:
        if volatility is None:
            raise ValueError("Волатильность не может быть None.")

        if volatility > 5:
            return default_threshold - 0.1
        elif volatility < 2:
            return default_threshold + 0.1
        else:
            return default_threshold
    except Exception as e:
        logger.error(f"Ошибка при корректировке порога: {str(e)}")
        return default_threshold
