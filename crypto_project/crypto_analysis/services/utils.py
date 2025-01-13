import logging
from crypto_analysis.models import CryptoPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_predictions(predictions, probabilities, crypto, dates):
    logger.info(f"Сохранение предсказаний для криптовалюты: {crypto}")

    if len(predictions) != len(probabilities) or len(predictions) != len(dates):
        logger.error(f"Несоответствие размеров списков. predictions={len(predictions)}, probabilities={len(probabilities)}, dates={len(dates)}")
        return

    saved_count = 0
    for i, prediction in enumerate(predictions):
        try:
            prediction_date = dates[i]
            CryptoPrediction.objects.update_or_create(
                cryptocurrency_pair=crypto,
                prediction_date=prediction_date,
                defaults={
                    "predicted_price_change": 1 if prediction else -1,
                    "probability_increase": probabilities[i],
                    "probability_decrease": 1 - probabilities[i],
                },
            )
            saved_count += 1
        except Exception as e:
            logger.error(f"Ошибка при сохранении предсказания для {crypto} на {dates[i]}: {str(e)}")

    logger.info(f"Общее количество успешно сохранённых предсказаний для {crypto}: {saved_count}")
