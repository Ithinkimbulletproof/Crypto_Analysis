import logging
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.moving_average_convergence_divergence import (
    moving_average_convergence_divergence as macd,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_indicators(df):
    try:
        if "close_price" not in df.columns:
            raise ValueError("Столбец 'close_price' отсутствует в DataFrame.")

        df["rsi"] = rsi(df["close_price"].tolist(), 14)
        df["macd"] = macd(df["close_price"].tolist(), 12, 26)
        return df
    except Exception as e:
        logger.error(f"Ошибка при добавлении индикаторов: {str(e)}")
        return df


def calculate_prediction_confidence(df, volatility_col, rsi_col=None, macd_col=None):
    try:
        if df is None or volatility_col not in df.columns:
            raise ValueError("Не найден указанный столбец волатильности.")

        confidence = 1.0

        df["confidence"] = df[volatility_col].apply(lambda x: max(0.1, 1.0 - (x / 10)))

        if rsi_col and rsi_col in df.columns:
            df["confidence"] = df.apply(
                lambda row: row["confidence"] * (1.0 - abs(row[rsi_col] - 50) / 100),
                axis=1,
            )

        if macd_col and macd_col in df.columns:
            df["confidence"] = df.apply(
                lambda row: row["confidence"]
                * (1.0 if abs(row[macd_col]) < 0.1 else 0.9),
                axis=1,
            )

        df["confidence"] = df["confidence"].clip(lower=0.1, upper=1.0)

        return df
    except Exception as e:
        logger.error(f"Ошибка при расчете уверенности: {str(e)}")
        return df


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
