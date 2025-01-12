import logging
from datetime import datetime, timedelta, timezone
from crypto_analysis.models import MarketData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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


def analyze_and_update():
    cryptocurrencies = ["BTC/USDT"]
    models = get_models()
    total_predictions = 0
    correct_predictions = 0
    for crypto in cryptocurrencies:
        try:
            logger.info(f"Обработка данных для {crypto}")
            data_all = (
                MarketData.objects.filter(cryptocurrency=crypto)
                .order_by("-date")
                .values()
            )
            if not data_all.exists():
                logger.warning(f"Нет данных для {crypto}. Пропускаем.")
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
                scaler = StandardScaler()
                smote = SMOTE(random_state=42)
                logger.info(
                    f"Масштабирование и ресэмплинг данных для {crypto} ({period})"
                )
                X_scaled, y_resampled = scale_and_resample_data(X, y, scaler, smote)
                logger.info(
                    f"Разделение данных на обучающую и тестовую выборки для {crypto} ({period})"
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled,
                    y_resampled,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_resampled,
                )
                total_predictions, correct_predictions, predictions, probabilities = (
                    train_and_evaluate_models(
                        models,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        total_predictions,
                        correct_predictions,
                    )
                )
                dates = X_test.index if hasattr(X_test, "index") else None
                if dates is not None and len(dates) == len(predictions) == len(
                    probabilities
                ):
                    save_predictions(predictions, probabilities, crypto, dates.tolist())
                else:
                    logger.error(
                        "Несоответствие размеров списков predictions, probabilities и dates."
                    )
        except Exception as e:
            logger.error(f"Ошибка обработки {crypto}: {str(e)}")
    log_overall_stats(total_predictions, correct_predictions)


def log_overall_stats(total_predictions, correct_predictions):
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(f"Общее количество предсказаний: {total_predictions}")
    logger.info(f"Количество правильных предсказаний: {correct_predictions}")
    logger.info(f"Точность модели: {accuracy:.2f}")
