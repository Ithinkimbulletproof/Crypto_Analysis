import logging
from crypto_analysis.models import CryptoPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_predictions(
    predictions,
    probabilities,
    crypto,
    dates,
    volatilities,
    rsi_values=None,
    macd_values=None,
):
    logger.info(f"Сохранение предсказаний для криптовалюты: {crypto}")
    if (
        len(predictions) != len(probabilities)
        or len(predictions) != len(dates)
        or len(predictions) != len(volatilities)
    ):
        logger.error(
            f"Несоответствие размеров списков. predictions={len(predictions)}, probabilities={len(probabilities)}, dates={len(dates)}, volatilities={len(volatilities)}"
        )
        return
    saved_records = []
    for i, prediction in enumerate(predictions):
        try:
            confidence = 1.0
            volatility = volatilities[i]
            confidence *= max(0.1, 1.0 - (volatility / 10))
            if rsi_values:
                rsi_value = rsi_values[i]
                confidence *= 1.0 - abs(rsi_value - 50) / 100
            if macd_values:
                macd_value = macd_values[i]
                confidence *= 1.0 if abs(macd_value) < 0.1 else 0.9
            confidence = max(0.1, min(1.0, confidence))
            saved_records.append(
                CryptoPrediction(
                    cryptocurrency_pair=crypto,
                    prediction_date=dates[i],
                    predicted_price_change=1 if prediction else -1,
                    probability_increase=probabilities[i],
                    probability_decrease=1 - probabilities[i],
                    confidence_level=confidence,
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
