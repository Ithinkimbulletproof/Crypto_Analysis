import os
import logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from ta.trend import SMAIndicator, CCIIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def fetch_data_from_csv(file_path="processed_data.csv"):
    if not os.path.exists(file_path):
        logger.warning(f"Файл {file_path} не найден.")
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        logger.warning(f"Файл {file_path} пуст.")
        return None
    return df


def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    df["lowest_low"] = df["low"].rolling(window=k_window).min()
    df["highest_high"] = df["high"].rolling(window=k_window).max()
    df["stoch_k"] = (
        100 * (df["close"] - df["lowest_low"]) / (df["highest_high"] - df["lowest_low"])
    )
    df["stoch_d"] = df["stoch_k"].rolling(window=d_window).mean()
    return df.drop(columns=["lowest_low", "highest_high"])


def apply_technical_analysis(df):
    start_time = datetime.now()
    df["SMA_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(df["close"], window=200).sma_indicator()

    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()

    df["CCI"] = CCIIndicator(df["high"], df["low"], df["close"], window=14).cci()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()

    df = calculate_stochastic_oscillator(df)

    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()

    logger.info(f"Время выполнения технического анализа: {datetime.now() - start_time}")
    return df


def enhance_data_processing(df):
    start_time = datetime.now()
    for lag in [1, 3, 7]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"RSI_lag_{lag}"] = df["RSI"].shift(lag)

    for window in [10, 30, 90]:
        df[f"SMA_{window}"] = SMAIndicator(df["close"], window=window).sma_indicator()

    df["close"] = df["close"].interpolate(method="linear", limit_direction="both")
    df["RSI"] = df["RSI"].interpolate(method="linear", limit_direction="both")

    for column in df.columns:
        if df[column].isnull().any():
            logger.warning(f"Пропущенные значения в столбце {column}. Заполняем их.")
            df[column] = df[column].fillna(method="ffill").fillna(method="bfill")

    logger.info(f"Время выполнения улучшения данных: {datetime.now() - start_time}")
    return df


def split_data_by_period(df):
    now = datetime.now(timezone.utc)
    dates_90_days_ago = now - timedelta(days=90)
    dates_180_days_ago = now - timedelta(days=180)
    dates_365_days_ago = now - timedelta(days=365)

    df_90 = df[df["date"] >= dates_90_days_ago]
    df_180 = df[df["date"] >= dates_180_days_ago]
    df_365 = df[df["date"] >= dates_365_days_ago]
    return df_90, df_180, df_365


def save_to_csv(
    df, cryptocurrency, period, file_path="processed_with_technical_analysis.csv"
):
    df["cryptocurrency"] = cryptocurrency
    df["period"] = period
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)

    logger.info(f"Данные сохранены в {file_path}")


def evaluate_model_performance(
    df, target_column="signal", prediction_column="predicted_signal"
):
    if target_column not in df.columns or prediction_column not in df.columns:
        logger.error("Не найдены целевые или предсказанные столбцы для анализа.")
        return

    y_true = df[target_column].dropna()
    y_pred = df[prediction_column].dropna()

    if len(y_true) != len(y_pred):
        logger.warning("Размеры реальных и предсказанных данных не совпадают.")
        return

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
        logger.info(f"ROC AUC: {roc_auc:.2f}")

        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        report = classification_report(
            y_true, y_pred, target_names=["Sell", "Hold", "Buy"]
        )
        logger.info(f"Classification Report:\n{report}")

    except Exception as e:
        logger.error(f"Ошибка анализа метрик: {str(e)}")


def generate_signals(df):
    try:
        df["signal"] = 0
        df.loc[df["RSI"] < 30, "signal"] = 1
        df.loc[df["RSI"] > 70, "signal"] = -1
        df["predicted_signal"] = df["signal"].shift(-1)
    except Exception as e:
        logger.error(f"Ошибка генерации сигналов: {str(e)}")
    return df


def process_and_evaluate_data():
    input_file_path = "processed_data.csv"
    df = fetch_data_from_csv(input_file_path)
    if df is None:
        return

    cryptocurrencies = df["cryptocurrency"].unique()

    for crypto in cryptocurrencies:
        try:
            logger.info(f"Обработка данных для {crypto}")

            df_crypto = df[df["cryptocurrency"] == crypto]

            last_modified = os.path.getmtime(input_file_path)
            last_update_date = datetime.fromtimestamp(last_modified)
            today = datetime.now()
            days_since_update = (today - last_update_date).days

            if days_since_update < 1:
                logger.info(
                    f"Данные для {crypto} недавно обновлены. Пропускаем обработку."
                )
                continue

            df_crypto = apply_technical_analysis(df_crypto)
            df_crypto = enhance_data_processing(df_crypto)
            df_crypto = generate_signals(df_crypto)

            evaluate_model_performance(df_crypto)
            save_to_csv(df_crypto, crypto, "evaluation")

        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")

    log_overall_stats()


def log_overall_stats():
    logger.info("Обработка завершена.")


if __name__ == "__main__":
    process_and_evaluate_data()
