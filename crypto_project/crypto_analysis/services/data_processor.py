import time
import logging
import pandas as pd
from pyti.smoothed_moving_average import smoothed_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_average_data(data_all):
    logger.info("Вычисление средних значений по всем биржам.")
    df = pd.DataFrame(list(data_all))
    logger.info(f"Типы данных в DataFrame:\n{df.dtypes}")
    logger.info(f"Первые несколько строк DataFrame:\n{df.head()}")
    if "date" not in df.columns:
        logger.error("Столбец 'date' отсутствует в DataFrame.")
        return None
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        logger.error(f"Ошибка при преобразовании столбца 'date': {str(e)}")
        return None
    unique_combinations = df[["date", "cryptocurrency"]].drop_duplicates()
    logger.info(
        f"Количество уникальных комбинаций (date, cryptocurrency): {len(unique_combinations)}"
    )
    logger.info(f"Количество записей до удаления дубликатов: {len(df)}")
    duplicates = df[df.duplicated(subset=["date", "cryptocurrency"], keep=False)]
    if not duplicates.empty:
        logger.warning("Обнаружены дубликаты в данных:")
        logger.warning(duplicates)
        df.drop_duplicates(subset=["date", "cryptocurrency"], inplace=True)
    logger.info(f"Количество записей после удаления дубликатов: {len(df)}")
    logger.info(f"Количество записей до удаления строк с NaN: {len(df)}")
    df.dropna(
        subset=["open_price", "high_price", "low_price", "close_price", "volume"],
        inplace=True,
    )
    logger.info(f"Количество записей после удаления строк с NaN: {len(df)}")
    unique_combinations_after_drop = df[["date", "cryptocurrency"]].drop_duplicates()
    logger.info(
        f"Количество уникальных комбинаций (date, cryptocurrency) после удаления дубликатов: {len(unique_combinations_after_drop)}"
    )
    avg_df = df.groupby(["date", "cryptocurrency"]).mean().reset_index()
    logger.info(f"После группировки: {avg_df.head()}")
    logger.info(f"Количество записей после группировки: {len(avg_df)}")
    group_sizes = (
        avg_df.groupby(["date", "cryptocurrency"]).size().reset_index(name="count")
    )
    logger.info(f"Размеры групп:\n{group_sizes}")
    if len(avg_df) < 14:
        logger.warning(f"Недостаточно данных для анализа. Пропускаем.")
        return None
    def calculate_sma_rsi(prices, window=14):
        if len(prices) >= window:
            sma_value = sma(prices[-window:], window)[-1]
            rsi_value = rsi(prices[-window:], window)[-1]
            return sma_value, rsi_value
        else:
            return None, None
    close_prices_dict = (
        avg_df.groupby("cryptocurrency")["close_price"].apply(list).to_dict()
    )
    avg_df["sma_14"], avg_df["rsi_14"] = zip(
        *avg_df.apply(
            lambda row: calculate_sma_rsi(close_prices_dict[row["cryptocurrency"]]),
            axis=1,
        )
    )
    if avg_df[["sma_14", "rsi_14"]].isnull().values.any():
        logger.warning(
            "В DataFrame присутствуют NaN значения после вычисления SMA и RSI."
        )
        logger.info(
            f"Столбцы с NaN значениями:\n{avg_df[['sma_14', 'rsi_14']].isnull().sum()}"
        )
        avg_df.dropna(subset=["sma_14", "rsi_14"], inplace=True)
    logger.info(f"Средние значения вычислены для {len(avg_df)} записей.")
    if len(avg_df) < 14:
        logger.warning(f"Недостаточно данных после вычисления SMA и RSI. Пропускаем.")
        return None
    if avg_df["sma_14"].nunique() == 1 and avg_df["rsi_14"].nunique() == 1:
        logger.warning(f"Некорректные значения SMA и RSI. Пропускаем.")
        return None
    return avg_df


def prepare_data_for_analysis(data_all):
    logger.info(f"Подготовка данных для анализа, количество записей: {len(data_all)}")
    df = pd.DataFrame(list(data_all))
    logger.info(f"Первые несколько строк DataFrame после загрузки:\n{df.head()}")
    avg_df = compute_average_data(df)
    if avg_df is None or len(avg_df) < 14:
        logger.warning(f"Недостаточно данных для анализа. Пропускаем.")
        return None
    avg_df = avg_df.fillna(method="ffill").fillna(method="bfill")
    logger.info(f"Количество записей в DataFrame до обработки: {len(avg_df)}")
    df = process_data(avg_df)
    if df is not None:
        logger.info(
            f"Подготовка данных завершена, количество записей после обработки: {len(df)}"
        )
    return df


def process_data(df):
    start_time = time.time()
    logger.info(f"Начало обработки данных: {len(df)} записей")
    if "close_price" not in df.columns or len(df["close_price"]) < 14:
        logger.warning("Недостаточно данных или отсутствует колонка 'close_price'.")
        return None
    close_prices = df["close_price"].tolist()
    df["SMA_14"] = sma(close_prices, 14)
    df["RSI_14"] = rsi(close_prices, 14)
    df["volatility"] = df["close_price"].rolling(window=14).std()
    df["volume_change"] = df["volume"].pct_change()
    df["price_change"] = df["close_price"].pct_change()
    ema_12 = df["close_price"].ewm(span=12).mean()
    ema_26 = df["close_price"].ewm(span=26).mean()
    df["MACD"] = ema_12 - ema_26
    df["EMA_9"] = df["close_price"].ewm(span=9).mean()
    df["momentum"] = df["close_price"].diff(4)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["hour_of_day"] = pd.to_datetime(df["date"]).dt.hour
    df["actual_change"] = (df["price_change"] > 0).astype(int)
    logger.info(f"NaN в столбцах перед обработкой:\n{df.isnull().sum()}")
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(
        subset=["SMA_14", "RSI_14", "volatility", "volume_change", "MACD"],
        inplace=True,
    )
    logger.info(f"Количество записей после обработки: {len(df)}")
    logger.info(f"NaN в столбцах после обработки:\n{df.isnull().sum()}")
    end_time = time.time()
    logger.info(f"Обработка данных завершена за {end_time - start_time:.2f} секунд.")
    return df


def get_features_and_labels(df):
    logger.info(f"Получение признаков и меток из данных: {len(df)} записей")
    feature_columns = [
        "close_price",
        "volume",
        "SMA_14",
        "RSI_14",
        "volatility",
        "volume_change",
        "MACD",
        "day_of_week",
        "hour_of_day",
    ]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют следующие столбцы: {missing_columns}")
    if "actual_change" not in df.columns:
        raise ValueError("Столбец 'actual_change' отсутствует в данных.")
    X = df[feature_columns].copy()
    y = df["actual_change"].copy()
    if X.isnull().any().any():
        raise ValueError("Признаки содержат пропущенные значения (NaN).")
    if y.isnull().any():
        raise ValueError("Целевая метка содержит пропущенные значения (NaN).")
    logger.info(f"Количество признаков: {X.shape[1]}, Количество меток: {len(y)}")
    return X.values, y.values


def scale_and_resample_data(X, y, scaler=None, smote=None):
    logger.info("Начало обработки данных для масштабирования и ресемплинга")
    X = pd.DataFrame(X)
    logger.info(f"Количество признаков до обработки: {X.shape[1]}")
    X = X.bfill().ffill()
    if X.isnull().any().any():
        logger.warning("Пропущенные значения все еще присутствуют. Удаляем строки.")
        X.dropna(inplace=True)
        y = y[: len(X)]
    logger.info(f"Данные после обработки пропусков: {X.shape}")
    positive_class_ratio = sum(y) / len(y)
    logger.info(
        f"Распределение классов: положительный класс - {positive_class_ratio:.2%}"
    )
    if positive_class_ratio < 0.1 or positive_class_ratio > 0.9:
        logger.warning(
            "Данные сильно дисбалансированы. Возможно, SMOTE не подходит для обработки."
        )
    if scaler is None:
        scaler = StandardScaler()
    logger.info("Масштабирование данных")
    X_scaled = scaler.fit_transform(X)
    if smote is None:
        smote = SMOTE()
    logger.info("Ресемплинг данных с использованием SMOTE")
    try:
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    except ValueError as e:
        logger.error(f"Ошибка при ресемплинге: {e}")
        return X_scaled, y
    logger.info(f"Размер данных после ресемплинга: {X_resampled.shape[0]} примеров")
    return X_resampled, y_resampled
