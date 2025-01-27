import time
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
        start_total_time = time.time()
        try:
            self.stdout.write(self.style.SUCCESS("Запуск получения данных..."))
            start_time = time.time()
            fetch_data()
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Данные успешно получены и сохранены за {time_used:.2f} секунд")
            )

            self.stdout.write(self.style.SUCCESS("Запуск обработки индикаторов..."))
            start_time = time.time()
            process_all_indicators()
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Все индикаторы успешно рассчитаны за {time_used:.2f} секунд")
            )

            self.stdout.write(self.style.SUCCESS("Запуск расчёта предсказаний..."))
            start_time = time.time()
            run_short_predictions()
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Все предсказания рассчитаны за {time_used:.2f} секунд")
            )

            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.SUCCESS(f"Полное время выполнения команды: {total_time:.2f} секунд")
            )

        except Exception as e:
            logger.error(f"Ошибка при выполнении команды: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.WARNING(f"Команда прервана после {total_time:.2f} секунд")
            )
