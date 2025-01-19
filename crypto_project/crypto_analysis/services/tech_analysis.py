import os
import logging
import numpy as np
import pandas as pd
from crypto_analysis.models import PreprocessedData, TechAnalysed
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


def fetch_data_from_db(cryptocurrency, period):
    period_number = period
    data = PreprocessedData.objects.filter(
        cryptocurrency=cryptocurrency, period=period_number
    ).values()

    if not data:
        logger.warning(
            f"Данные для {cryptocurrency} в периоде {period_number} не найдены."
        )
        return None

    try:
        if isinstance(data, list):
            df = pd.DataFrame.from_records(data)
        else:
            df = pd.DataFrame(data)

        if df.empty:
            logger.warning(
                f"Создан пустой DataFrame для {cryptocurrency} в периоде {period}."
            )
            return None

        logger.info(f"Получены данные для {cryptocurrency} в периоде {period} дней")
    except Exception as e:
        logger.error(f"Ошибка создания DataFrame: {str(e)}")
        return None

    return df


def check_data_quality(df, stage="initial"):
    missing_data = df.isnull().sum()
    if missing_data.any():
        logger.warning(f"Пропущенные значения на стадии {stage}:\n{missing_data[missing_data > 0]}")

    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        anomalies = df[(df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)]
        if not anomalies.empty:
            logger.warning(f"Аномалии на стадии {stage} в столбце {col}:\n{anomalies}")

    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        logger.warning(f"Дублированные строки на стадии {stage}:\n{duplicates}")

def check_required_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Отсутствуют обязательные столбцы: {missing_columns}")
        return False
    return True


def split_data_by_period(df, periods=[30, 90, 180, 365]):
    now = datetime.now(timezone.utc)
    split_data = {}

    df["date"] = pd.to_datetime(df["date"])

    for period in periods:
        date_limit = now - timedelta(days=period)
        split_data[period] = df[df["date"] >= date_limit]

    return split_data


def enhance_data_processing(df, imputer, logger):
    start_time = datetime.now()

    for lag in [1, 3, 7, 30, 90]:
        df[f"close_lag_{lag}"] = df["close_price"].shift(lag)
        if "RSI" in df.columns:
            df[f"RSI_lag_{lag}"] = df["RSI"].shift(lag)

    for window in [10, 30, 90]:
        df[f"SMA_{window}"] = SMAIndicator(
            df["close_price"], window=window
        ).sma_indicator()

    missing_before = df.isnull().sum()
    logger.info(
        f"Пропущенные значения перед обработкой:\n{missing_before[missing_before > 0]}"
    )

    numerics_columns = [
        "close_price",
        "high_price",
        "low_price",
        "SMA_50",
        "SMA_200",
        "RSI",
        "CCI",
        "atr",
        "bb_bbm",
        "bb_bbh",
        "bb_bbl",
        "macd_diff",
    ]
    numerics_columns = [col for col in numerics_columns if col in df.columns]

    df[numerics_columns] = df[numerics_columns].astype("float64")

    df = df.interpolate(method="linear", limit_direction="both")
    df = df.fillna(method="ffill").fillna(method="bfill")

    try:
        if numerics_columns:
            df[numerics_columns] = imputer.fit_transform(df[numerics_columns])
    except Exception as e:
        logger.error(f"Ошибка импутирования: {str(e)}")

    missing_after = df.isnull().sum()
    logger.info(
        f"Пропущенные значения после обработки:\n{missing_after[missing_after > 0]}"
    )
    logger.info(f"Время выполнения улучшения данных: {datetime.now() - start_time}")
    return df


def generate_target_variable(df, period=24):
    df["target"] = 0
    df["future_close"] = df["close"].shift(-period)

    df.loc[df["future_close"] > df["close"], "target"] = 1
    df.loc[df["future_close"] < df["close"], "target"] = -1

    df.drop(columns=["future_close"], inplace=True)
    return df


def apply_technical_analysis(df):
    start_time = datetime.now()

    for lag in [1, 3, 7, 30, 90]:
        if len(df) > lag:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"RSI_lag_{lag}"] = (
                df["RSI"].shift(lag) if "RSI" in df.columns else np.nan
            )

    df_copy = df.copy()

    df_copy["close"] = pd.to_numeric(df_copy["close_price"], errors="coerce")
    df_copy["high"] = pd.to_numeric(df_copy["high_price"], errors="coerce")
    df_copy["low"] = pd.to_numeric(df_copy["low_price"], errors="coerce")

    logger.info("Проверка наличия столбцов для анализа: 'close', 'high', 'low'")
    if not all(col in df_copy.columns for col in ["close", "high", "low"]):
        logger.error("Отсутствуют необходимые столбцы для расчёта индикаторов.")
        return df_copy

    try:
        if "SMA_50" not in df_copy.columns:
            df_copy["SMA_50"] = SMAIndicator(
                df_copy["close"], window=50
            ).sma_indicator()
        if "SMA_200" not in df_copy.columns:
            df_copy["SMA_200"] = SMAIndicator(
                df_copy["close"], window=200
            ).sma_indicator()

        if "RSI" not in df_copy.columns:
            df_copy["RSI"] = RSIIndicator(df_copy["close"], window=14).rsi()

        df_copy["CCI"] = CCIIndicator(
            df_copy["high"], df_copy["low"], df_copy["close"], window=14
        ).cci()
        atr = AverageTrueRange(
            df_copy["high"], df_copy["low"], df_copy["close"], window=14
        )
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
        "close",
        "high",
        "low",
        "SMA_50",
        "SMA_200",
        "RSI",
        "CCI",
        "atr",
        "bb_bbm",
        "bb_bbh",
        "bb_bbl",
        "macd_diff",
    ]
    numerics_columns = [col for col in numerics_columns if col in df_copy.columns]

    try:
        df_copy[numerics_columns] = imputer.fit_transform(df_copy[numerics_columns])
    except Exception as e:
        logger.error(f"Ошибка импутирования индикаторов: {str(e)}")

    df_copy["predicted_signal"] = 0
    df_copy.loc[df_copy["SMA_50"] > df_copy["SMA_200"], "predicted_signal"] = 1
    df_copy.loc[df_copy["SMA_50"] < df_copy["SMA_200"], "predicted_signal"] = -1

    logger.info(f"Время выполнения технического анализа: {datetime.now() - start_time}")
    return df_copy


def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    required_columns = ["low", "high", "close"]

    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Столбец '{col}' отсутствует в данных")

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


def save_to_db(df, cryptocurrency, period):
    tech_analysed_objects = []
    for _, row in df.iterrows():
        tech_analysed = TechAnalysed(
            date=row["date"],
            cryptocurrency=cryptocurrency,
            period=period,
            close_price=row["close"],
            high_price=row["high"],
            low_price=row["low"],
            price_change_24h=row.get("price_change_24h", None),
            SMA_30=row.get("SMA_30", None),
            volatility_30=row.get("volatility_30", None),
            SMA_90=row.get("SMA_90", None),
            volatility_90=row.get("volatility_90", None),
            SMA_180=row.get("SMA_180", None),
            volatility_180=row.get("volatility_180", None),
            predicted_signal=row.get("predicted_signal", None),
            target=row.get("target", None),
        )
        tech_analysed_objects.append(tech_analysed)

    TechAnalysed.objects.bulk_create(tech_analysed_objects)
    logger.info(f"Данные для {cryptocurrency} в периоде {period} успешно сохранены.")


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
    cryptocurrencies = os.getenv("CRYPTOPAIRS").split(",")
    periods = [30, 90, 180, 365]
    for crypto in cryptocurrencies:
        for period in periods:
            try:
                logger.info(f"Обработка данных для {crypto} в периоде {period} дней")
                df_crypto = fetch_data_from_db(cryptocurrency=crypto, period=period)
                if df_crypto is None:
                    continue

                if not check_required_columns(df_crypto, ["close", "high", "low"]):
                    continue

                check_data_quality(df_crypto, stage="fetch")

                split_data = split_data_by_period(df_crypto, periods=[period])
                df_crypto = split_data[period]

                check_data_quality(df_crypto, stage="split")

                if not check_required_columns(df_crypto, ["close", "high", "low"]):
                    continue

                df_crypto = apply_technical_analysis(df_crypto)

                check_data_quality(df_crypto, stage="technical_analysis")

                df_crypto = enhance_data_processing(df_crypto)

                check_data_quality(df_crypto, stage="enhanced_processing")

                df_crypto = generate_target_variable(df_crypto, period=24)

                if not check_required_columns(df_crypto, ["close", "high", "low"]):
                    continue

                df_crypto = apply_technical_analysis(df_crypto)

                check_data_quality(df_crypto, stage="final")

                evaluate_model_performance(df_crypto)
                save_to_db(df_crypto, crypto, period)
            except Exception as e:
                logger.error(f"Ошибка обработки {crypto} в периоде {period}: {str(e)}")
    log_overall_stats()


def log_overall_stats():
    logger.info("Обработка завершена.")


if __name__ == "__main__":
    process_and_evaluate_data()
