import os
import logging
import numpy as np
import pandas as pd
from ta.trend import MACD
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, CCIIndicator
from datetime import datetime, timedelta, timezone
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

numeric_columns = ["close", "high", "low", "RSI", "SMA_50", "SMA_200"]
imputer = IterativeImputer(max_iter=10, random_state=42)


def fetch_data_from_csv(file_path="processed_data.csv"):
    if not os.path.exists(file_path):
        logger.warning(f"Файл {file_path} не найден.")
        return None
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        logger.warning(f"Файл {file_path} пуст.")
        return None

    missing_data = df.isnull().sum()
    logger.info(
        f"Пропущенные значения в исходных данных:\n{missing_data[missing_data > 0]}"
    )

    return df


def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["low", "high", "close"]).copy()
    df.loc[:, "lowest_low"] = df["low"].rolling(window=k_window).min()
    df.loc[:, "highest_high"] = df["high"].rolling(window=k_window).max()
    df.loc[:, "stoch_k"] = (
            100 * (df["close"] - df["lowest_low"]) / (df["highest_high"] - df["lowest_low"])
    )
    df.loc[:, "stoch_d"] = df["stoch_k"].rolling(window=d_window).mean()

    df = df.dropna(subset=["stoch_k", "stoch_d"])

    return df.drop(columns=["lowest_low", "highest_high"])


def apply_technical_analysis(df):
    start_time = datetime.now()

    df_copy = df.copy()

    df_copy["close"] = pd.to_numeric(df_copy["close"], errors="coerce")
    df_copy["high"] = pd.to_numeric(df_copy["high"], errors="coerce")
    df_copy["low"] = pd.to_numeric(df_copy["low"], errors="coerce")

    logger.info("Проверка наличия столбцов для анализа: 'close', 'high', 'low'")
    if not all(col in df_copy.columns for col in ["close", "high", "low"]):
        logger.error("Отсутствуют необходимые столбцы для расчёта индикаторов.")
        return df_copy

    try:
        if "SMA_50" not in df_copy.columns:
            df_copy["SMA_50"] = SMAIndicator(df_copy["close"], window=50).sma_indicator()
        if "SMA_200" not in df_copy.columns:
            df_copy["SMA_200"] = SMAIndicator(df_copy["close"], window=200).sma_indicator()

        if "RSI" not in df_copy.columns:
            df_copy["RSI"] = RSIIndicator(df_copy["close"], window=14).rsi()

        df_copy["CCI"] = CCIIndicator(df_copy["high"], df_copy["low"], df_copy["close"], window=14).cci()
        atr = AverageTrueRange(df_copy["high"], df_copy["low"], df_copy["close"], window=14)
        df_copy["atr"] = atr.average_true_range()
        bb = BollingerBands(df_copy["close"], window=20, window_dev=2)
        df_copy["bb_bbm"] = bb.bollinger_mavg()
        df_copy["bb_bbh"] = bb.bollinger_hband()
        df_copy["bb_bbl"] = bb.bollinger_lband()

        df_copy = calculate_stochastic_oscillator(df_copy)

        macd = MACD(df_copy["close"])
        df_copy["macd_diff"] = macd.macd_diff()

    except Exception as e:
        logger.error(f"Ошибка при расчете индикаторов: {str(e)}")
        return df_copy

    numerics_columns = [
        "close", "high", "low", "SMA_50", "SMA_200", "RSI", "CCI", "atr",
        "bb_bbm", "bb_bbh", "bb_bbl", "macd_diff"
    ]

    numerics_columns = [col for col in numerics_columns if col in df_copy.columns]

    df_copy[numerics_columns] = imputer.fit_transform(df_copy[numerics_columns])

    logger.info(f"Пропущенные значения после обработки данных:\n{df_copy.isnull().sum()}")

    df_copy["predicted_signal"] = 0
    df_copy.loc[df_copy["SMA_50"] > df_copy["SMA_200"], "predicted_signal"] = 1
    df_copy.loc[df_copy["SMA_50"] < df_copy["SMA_200"], "predicted_signal"] = -1

    logger.info(f"Время выполнения технического анализа: {datetime.now() - start_time}")
    return df_copy


def enhance_data_processing(df):
    start_time = datetime.now()

    for lag in [1, 3, 7, 30, 90]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"RSI_lag_{lag}"] = df["RSI"].shift(lag) if "RSI" in df.columns else np.nan

    for window in [10, 30, 90]:
        df[f"SMA_{window}"] = SMAIndicator(df["close"], window=window).sma_indicator()

    missing_before = df.isnull().sum()
    logger.info(
        f"Пропущенные значения перед обработкой:\n{missing_before[missing_before > 0]}"
    )

    df = df.infer_objects()

    df = df.interpolate(method="linear", limit_direction="both")

    missing_after = df.isnull().sum()
    logger.info(
        f"Пропущенные значения после обработки:\n{missing_after[missing_after > 0]}"
    )

    logger.info(f"Время выполнения улучшения данных: {datetime.now() - start_time}")
    return df


def split_data_by_period(df, periods=[90, 180, 365]):
    now = datetime.now(timezone.utc)
    split_data = {}

    df["date"] = pd.to_datetime(df["date"])

    for period in periods:
        date_limit = now - timedelta(days=period)
        split_data[period] = df[df["date"] >= date_limit]

    return split_data


def generate_target_variable(df, period=24):
    df["target"] = 0

    df.loc[df["close"].shift(-period) > df["close"], "target"] = 1

    df.loc[df["close"].shift(-period) < df["close"], "target"] = -1

    df["target"] = df["target"].astype(int)
    return df


def save_to_csv(
    df, cryptocurrency, period, file_path="processed_with_technical_analysis.csv"
):
    missing_data_before_save = df.isnull().sum()
    logger.info(
        f"Пропущенные значения перед сохранением:\n{missing_data_before_save[missing_data_before_save > 0]}"
    )

    df["cryptocurrency"] = cryptocurrency
    df["period"] = period
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)

    logger.info(f"Данные сохранены в {file_path}")


def evaluate_model_performance(
    df, target_column="target", prediction_column="predicted_signal"
):
    if target_column not in df.columns or prediction_column not in df.columns:
        logger.error(
            f"Не найдены целевые или предсказанные столбцы для анализа: {target_column}, {prediction_column}"
        )
        return

    y_true = df[target_column].dropna()
    y_pred = df[prediction_column].dropna()

    if len(y_true) != len(y_pred):
        logger.warning(
            f"Размеры целевых ({len(y_true)}) и предсказанных данных ({len(y_pred)}) не совпадают."
        )
        return

    y_pred = np.where(y_pred == 1, 1, 0)
    y_true = np.where(y_true > 0, 1, 0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
        logger.info(f"ROC AUC: {roc_auc:.2f}")

        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        report = classification_report(y_true, y_pred, target_names=["Sell", "Buy"])
        logger.info(f"Classification Report:\n{report}")

    except Exception as e:
        logger.error(f"Ошибка анализа метрик: {str(e)}")


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

            required_columns = ["close", "high", "low"]
            if not all(col in df_crypto.columns for col in required_columns):
                logger.error(f"Отсутствуют обязательные столбцы для {crypto}: {required_columns}")
                continue

            df_crypto = apply_technical_analysis(df_crypto)
            df_crypto = enhance_data_processing(df_crypto)
            df_crypto = generate_target_variable(df_crypto, period=24)

            evaluate_model_performance(df_crypto)
            save_to_csv(df_crypto, crypto, "evaluation")

        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")

    log_overall_stats()


def log_overall_stats():
    logger.info("Обработка завершена.")


if __name__ == "__main__":
    process_and_evaluate_data()
