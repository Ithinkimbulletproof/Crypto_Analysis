import time
from django.core.management.base import BaseCommand
from crypto_analysis.services.data_fetcher import fetch_data
from crypto_analysis.services.data_preprocessing import process_all_indicators
from crypto_analysis.services.indicators_only import predict_for_all_cryptos
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Получает данные с бирж, сохраняет их в базе данных, рассчитывает индикаторы и предсказывает изменения"

    def handle(self, *args, **kwargs):
        start_time = time.time()

        try:
            self.stdout.write(self.style.SUCCESS("Запуск получения данных..."))
            fetch_data()
            self.stdout.write(self.style.SUCCESS("Данные успешно получены и сохранены"))

            self.stdout.write(self.style.SUCCESS("Запуск обработки индикаторов..."))
            process_all_indicators()
            self.stdout.write(
                self.style.SUCCESS("Все индикаторы успешно рассчитаны и сохранены")
            )

            self.stdout.write(self.style.SUCCESS("Запуск предсказательной модели..."))
            predict_for_all_cryptos()
            self.stdout.write(self.style.SUCCESS("Предсказания успешно выполнены"))

        except Exception as e:
            logger.error(f"Ошибка при выполнении команды: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.stdout.write(
            self.style.SUCCESS(f"Время выполнения: {elapsed_time:.2f} секунд")
        )
