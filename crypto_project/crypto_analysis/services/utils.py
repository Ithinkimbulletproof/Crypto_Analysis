import logging
import numpy as np
from crypto_analysis.models import CryptoPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_predictions(predictions, probabilities, crypto, dates):
    logger.info(f"Сохранение предсказаний для криптовалюты: {crypto}")
    if hasattr(dates, "to_list"):
        dates = dates.to_list()
    elif isinstance(dates, np.ndarray):
        dates = dates.tolist()
    elif not isinstance(dates, list):
        logger.error(
            "Формат дат неподдерживаемый. Ожидается list, numpy array или pandas Series."
        )
        return
    if len(predictions) != len(probabilities) or len(predictions) != len(dates):
        logger.error(
            "Несоответствие размеров списков predictions, probabilities и dates."
        )
        return
    for i, prediction in enumerate(predictions):
        try:
            prediction_date = dates[i]
            logger.info(f"Сохранение предсказания для даты: {prediction_date}")
            CryptoPrediction.objects.update_or_create(
                cryptocurrency_pair=crypto,
                prediction_date=prediction_date,
                defaults={
                    "predicted_price_change": 1 if prediction else -1,
                    "probability_increase": probabilities[i],
                    "probability_decrease": 1 - probabilities[i],
                },
            )
            logger.info(
                f"Предсказание для {crypto} на {prediction_date} успешно сохранено."
            )
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении предсказания для {crypto} на {dates[i]}: {str(e)}"
            )
