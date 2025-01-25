import logging
from django.core.management.base import BaseCommand
from crypto_analysis.services.data_fetcher import fetch_data
from crypto_analysis.services.data_preprocessing import process_all_indicators
from crypto_analysis.services.short_predictions import run_short_predictions

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Получает данные с бирж, сохраняет их в базе данных и рассчитывает индикаторы"
    )

    def handle(self, *args, **kwargs):
        try:
            self.stdout.write(self.style.SUCCESS("Запуск получения данных..."))
            fetch_data()
            self.stdout.write(self.style.SUCCESS("Данные успешно получены и сохранены"))

            self.stdout.write(self.style.SUCCESS("Запуск обработки индикаторов..."))
            process_all_indicators()
            self.stdout.write(
                self.style.SUCCESS("Все индикаторы успешно рассчитаны и сохранены"))
            self.stdout.write(self.style.SUCCESS("Запуск расчёта предсказаний..."))
            run_short_predictions()
            self.style.SUCCESS("Все предсказания успешно рассчитаны и сохранены")
        except Exception as e:
            logger.error(f"Ошибка при выполнении команды: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
