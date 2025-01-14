import logging
from crypto_analysis.models import CryptoPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_predictions(predictions, probabilities, crypto, dates):
    logger.info(f"Сохранение предсказаний для криптовалюты: {crypto}")
    if len(predictions) != len(probabilities) or len(predictions) != len(dates):
        logger.error(
            f"Несоответствие размеров списков. predictions={len(predictions)}, probabilities={len(probabilities)}, dates={len(dates)}"
        )
        return
    saved_records = []
    for i, prediction in enumerate(predictions):
        try:
            saved_records.append(
                CryptoPrediction(
                    cryptocurrency_pair=crypto,
                    prediction_date=dates[i],
                    predicted_price_change=1 if prediction else -1,
                    probability_increase=probabilities[i],
                    probability_decrease=1 - probabilities[i],
                )
            )
        except Exception as e:
            logger.error(
                f"Ошибка при создании записи для {crypto} на {dates[i]}: {str(e)}"
            )
    if saved_records:
        try:
            CryptoPrediction.objects.bulk_create(saved_records, ignore_conflicts=True)
            logger.info(
                f"Общее количество успешно сохранённых предсказаний для {crypto}: {len(saved_records)}"
            )
        except Exception as e:
            logger.error(f"Ошибка при массовом сохранении предсказаний: {str(e)}")
