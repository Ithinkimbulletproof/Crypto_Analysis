import os
import logging
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from crypto_analysis.models import MarketData
from sklearn.model_selection import train_test_split
from crypto_analysis.services.data_processor import (
    prepare_data_for_analysis,
    get_features_and_labels,
    scale_and_resample_data,
)
from crypto_analysis.services.machine_learning import (
    train_and_evaluate_models,
    get_models,
)
from crypto_analysis.services.utils import save_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def analyze_and_update():
    cryptocurrencies = os.getenv("CRYPTOPAIRS").split(",")
    models = get_models()
    total_predictions = 0
    correct_predictions = 0
    for crypto in cryptocurrencies:
        try:
            logger.info(f"Обработка данных для {crypto}")
            data_all = (
                MarketData.objects.filter(cryptocurrency=crypto)
                .order_by("date")
                .values()
            )
            if not data_all.exists():
                logger.warning(f"Нет данных для {crypto}. Пропускаем.")
                continue
            data_all = list(data_all)
            logger.debug(f"Пример входных данных: {data_all[:3]}")
            if not data_all or not all(isinstance(d, dict) for d in data_all):
                logger.error(
                    f"Некорректный формат данных для {crypto}. Ожидается список словарей."
                )
                continue
            df = prepare_data_for_analysis(data_all)
            if df is None:
                logger.warning(
                    f"Не удалось подготовить данные для анализа {crypto}. Пропускаем."
                )
                continue
            logger.info(f"Количество записей после подготовки: {len(df)}")
            now = datetime.now(timezone.utc)
            dates_90_days_ago = now - timedelta(days=90)
            dates_180_days_ago = now - timedelta(days=180)
            dates_365_days_ago = now - timedelta(days=365)
            df_90 = df[df["date"] >= dates_90_days_ago]
            df_180 = df[df["date"] >= dates_180_days_ago]
            df_365 = df[df["date"] >= dates_365_days_ago]
            logger.info(
                f"Подготовка данных завершена для {crypto}, начинаем обучение моделей."
            )
            for period, df_period in zip(
                ["90_days", "180_days", "365_days"], [df_90, df_180, df_365]
            ):
                logger.info(f"Обработка периода {period}")
                logger.info(f"Количество записей для {period}: {len(df_period)}")
                if len(df_period) < 14:
                    logger.warning(
                        f"Недостаточно данных для периода {period}. Пропускаем."
                    )
                    continue
                X, y = get_features_and_labels(df_period)
                if X is None or y is None:
                    logger.error(
                        f"Не удалось получить признаки и метки для {crypto} ({period}). Пропускаем."
                    )
                    continue
                logger.debug(f"Пример признаков: {X[:3]}")
                logger.debug(f"Пример меток: {y[:3]}")
                logger.info(
                    f"Масштабирование и ресэмплирование данных для {crypto} ({period})"
                )
                X_resampled, y_resampled, original_indices_resampled = (
                    scale_and_resample_data(X, y)
                )
                if (
                    X_resampled is None
                    or y_resampled is None
                    or original_indices_resampled is None
                ):
                    logger.error(
                        f"Не удалось масштабировать и ресемплировать данные для {crypto} ({period}). Пропускаем."
                    )
                    continue
                logger.info(
                    f"Разделение данных на обучающую и тестовую выборки для {crypto} ({period})"
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled,
                    y_resampled,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_resampled,
                )
                test_indices = np.arange(len(X_resampled))[
                    ~np.isin(np.arange(len(X_resampled)), np.arange(len(X_train)))
                ]
                test_original_indices = original_indices_resampled[test_indices]
                test_original_indices = test_original_indices[
                    test_original_indices != -1
                ]
                if not all(test_original_indices < len(df_period)):
                    logger.error(
                        f"Индексы тестовой выборки вышли за пределы DataFrame: {test_original_indices}"
                    )
                    continue
                dates = df_period.iloc[test_original_indices].date.tolist()
                logger.info(f"Количество дат: {len(dates)}")
                total_predictions, correct_predictions, predictions, probabilities = (
                    train_and_evaluate_models(
                        models, X_train, y_train, X_test, y_test, df_period
                    )
                )
                logger.info(f"Количество предсказаний: {len(predictions)}")
                logger.info(f"Количество вероятностей: {len(probabilities)}")
                if len(predictions) == len(probabilities) == len(dates):
                    latest_prediction = predictions[-1]
                    latest_probability = probabilities[-1]
                    latest_date = dates[-1]
                    save_predictions(
                        [latest_prediction], [latest_probability], crypto, [latest_date]
                    )
                else:
                    min_length = min(len(predictions), len(probabilities), len(dates))
                    predictions = predictions[:min_length]
                    probabilities = probabilities[:min_length]
                    dates = dates[:min_length]
                    logger.warning(f"Список данных обрезан до размера {min_length}")
                    latest_prediction = predictions[-1]
                    latest_probability = probabilities[-1]
                    latest_date = dates[-1]
                    save_predictions(
                        [latest_prediction], [latest_probability], crypto, [latest_date]
                    )
        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")
    log_overall_stats(total_predictions, correct_predictions)


def log_overall_stats(total_predictions, correct_predictions):
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(f"Общее количество предсказаний: {total_predictions}")
    logger.info(f"Количество правильных предсказаний: {correct_predictions}")
    logger.info(f"Точность модели: {accuracy:.2f}")
