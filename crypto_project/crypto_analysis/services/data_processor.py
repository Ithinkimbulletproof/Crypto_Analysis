import logging
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, CCIIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_indicators(df, window=14):
    try:
        if df.empty:
            logger.warning("Данные для расчета индикаторов пусты.")
            return None
        logger.info(f"Расчет индикаторов с окном {window}.")

        sma_indicator = SMAIndicator(close=df["close_price"], window=window)
        rsi_indicator = RSIIndicator(close=df["close_price"], window=window)
        cci_indicator = CCIIndicator(
            high=df["high_price"],
            low=df["low_price"],
            close=df["close_price"],
            window=window,
            constant=0.015,
        )
        df["SMA_14"] = sma_indicator.sma_indicator()
        df["RSI_14"] = rsi_indicator.rsi()
        df["CCI_14"] = cci_indicator.cci()
        df["volatility"] = df["close_price"].rolling(window=window).std()
        df["volume_change"] = df["volume"].pct_change()
        df["price_change"] = df["close_price"].pct_change()
        ema_12 = df["close_price"].ewm(span=12).mean()
        ema_26 = df["close_price"].ewm(span=26).mean()
        df["MACD"] = ema_12 - ema_26
        df["Signal_Line"] = df["MACD"].ewm(span=9).mean()
        df["momentum"] = df["close_price"].diff(4)
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["hour_of_day"] = pd.to_datetime(df["date"]).dt.hour

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Пропущенные значения после заполнения:\n{missing_values}")

    except KeyError as e:
        logger.error(f"Отсутствуют необходимые столбцы: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при расчете индикаторов: {str(e)}")
        return None
    return df


def compute_average_data(data_all, window=14):
    logger.info("Вычисление средних значений по всем биржам.")

    try:
        if (
            not data_all
            or not isinstance(data_all, list)
            or not all(isinstance(record, dict) for record in data_all)
        ):
            logger.error("Некорректный формат данных. Ожидается список словарей.")
            return None

        df = pd.DataFrame(data_all)

        required_columns = {
            "date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "cryptocurrency",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"Отсутствуют обязательные столбцы: {missing_columns}")
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        numeric_columns = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=numeric_columns, inplace=True)
        if df.empty:
            logger.warning("Все данные удалены после обработки пропусков.")
            return None
        logger.info(f"Количество строк после удаления пропусков: {len(df)}")

        if df["cryptocurrency"].nunique() > 1 or df["date"].duplicated().any():
            avg_df = (
                df.groupby(["date", "cryptocurrency"])
                .mean(numeric_only=True)
                .reset_index()
            )
        else:
            avg_df = df.copy()

        if len(avg_df) < window:
            logger.warning(
                f"Недостаточно данных для анализа (требуется минимум {window} записей). Попробуйте уменьшить окно анализа."
            )
            return None
        logger.info(f"Количество строк после группировки: {len(avg_df)}")

        avg_df = calculate_indicators(avg_df, window=window)
        if avg_df is None or avg_df.empty:
            logger.error("Ошибка при расчёте индикаторов. Пустой результат.")
            return None
        logger.info(f"Количество строк после расчёта индикаторов: {len(avg_df)}")

        avg_df.dropna(subset=["SMA_14", "RSI_14", "CCI_14"], inplace=True)
        if len(avg_df) < window:
            logger.warning(
                f"Недостаточно данных после расчёта индикаторов (требуется минимум {window} записей)."
            )
            return None
        logger.info(
            f"Количество строк после удаления пропусков индикаторов: {len(avg_df)}"
        )
        logger.debug(f"Первые строки данных после обработки: {avg_df.head()}")

    except Exception as e:
        logger.error(f"Ошибка при вычислении средних значений: {str(e)}")
        return None
    return avg_df


def prepare_data_for_analysis(data_all, window=14):
    logger.info(f"Подготовка данных для анализа: {len(data_all)} записей.")

    try:
        if (
            not data_all
            or not isinstance(data_all, list)
            or not all(isinstance(d, dict) for d in data_all)
        ):
            logger.warning(
                "Переданы некорректные или пустые данные. Ожидается список словарей."
            )
            return None

        logger.debug(f"Пример входных данных: {data_all[:3]}")

        required_keys = {
            "date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "cryptocurrency",
        }
        for record in data_all:
            missing_keys = required_keys - record.keys()
            if missing_keys:
                logger.error(f"Отсутствуют обязательные ключи в записи: {missing_keys}")
                return None

        avg_df = compute_average_data(data_all, window=window)
        if avg_df is None:
            logger.error("Ошибка подготовки данных.")
            return None

        logger.info(
            f"Количество строк после вычисления средних значений: {len(avg_df)}"
        )

        avg_df.ffill(inplace=True)
        avg_df.bfill(inplace=True)
        if avg_df.isnull().any().any():
            logger.warning("Остались пропуски после заполнения. Удаляем строки с NaN.")
            avg_df.dropna(inplace=True)

        logger.info(f"Количество строк после заполнения пропусков: {len(avg_df)}")

        if len(avg_df) < window:
            logger.warning(
                f"Недостаточно данных для анализа после обработки (требуется минимум {window} строк)."
            )
            return None

        logger.debug(f"Первые строки подготовленных данных: {avg_df.head()}")

    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {str(e)}")
        return None

    return avg_df


def process_data(df, window=14):
    if df is None or df.empty:
        logger.warning("Пустые данные переданы в process_data.")
        return None
    logger.info(f"Обработка данных: {len(df)} записей.")
    df = calculate_indicators(df, window=window)
    if df is None:
        return None

    logger.info(f"Количество строк после расчета индикаторов: {len(df)}")

    df.dropna(
        subset=["SMA_14", "RSI_14", "CCI_14", "volatility", "volume_change", "MACD"],
        inplace=True,
    )
    if df.empty:
        logger.warning("Все данные удалены после удаления пропусков.")
        return None

    logger.info(f"Количество строк после удаления пропусков: {len(df)}")

    return df


def get_features_and_labels(df):
    feature_columns = [
        "close_price",
        "volume",
        "SMA_14",
        "RSI_14",
        "CCI_14",
        "volatility",
        "volume_change",
        "MACD",
        "day_of_week",
        "hour_of_day",
    ]
    try:
        if not set(feature_columns).issubset(df.columns):
            missing_columns = set(feature_columns) - set(df.columns)
            raise ValueError(
                f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}"
            )
        X = df[feature_columns]
        y = df["price_change"].apply(lambda x: 1 if x > 0 else 0)
    except ValueError as e:
        logger.error(f"Ошибка при подготовке признаков и меток: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при подготовке признаков и меток: {str(e)}")
        return None, None
    return X.values, y.values


def scale_and_resample_data(X, y):
    try:
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Распределение классов перед SMOTE: {class_distribution}")
        min_samples_per_class = min(counts)
        if min_samples_per_class < 6:
            logger.warning(
                "Недостаточно данных для SMOTE. Возвращаем оригинальные данные."
            )
            return X, y, np.arange(len(X))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(k_neighbors=min(min_samples_per_class - 1, 5), random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        original_indices = np.arange(len(X))
        smote_indices = np.arange(len(X_resampled))
        original_indices_resampled = np.concatenate([
            original_indices,
            np.full(len(X_resampled) - len(X), -1)
        ])

        return X_resampled, y_resampled, original_indices_resampled
    except Exception as e:
        logger.error(f"Ошибка при масштабировании и ресемплинге данных: {str(e)}")
        return None, None, None


def main(data_all, window=14):
    logger.info("Запуск анализа данных.")
    avg_df = prepare_data_for_analysis(data_all, window=window)
    if avg_df is None:
        logger.error("Ошибка подготовки данных.")
        return None
    processed_df = process_data(avg_df, window=window)
    if processed_df is None:
        logger.error("Ошибка обработки данных.")
        return None
    X, y = get_features_and_labels(processed_df)
    if X is None or y is None:
        return None
    X_resampled, y_resampled = scale_and_resample_data(X, y)
    if X_resampled is None or y_resampled is None:
        return None
    logger.info("Анализ данных завершен.")
    return X_resampled, y_resampled
