import time
import asyncio
import logging
from django.core.management.base import BaseCommand
from crypto_analysis.services.data_fetcher import fetch_data
from crypto_analysis.services.data_preprocessing import process_all_indicators
from crypto_analysis.services.news_analysis import gather_and_analyze_news
from crypto_analysis.services.short_predictions import run_short_predictions

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Получает данные с бирж, сохраняет их в базе данных и рассчитывает индикаторы"
    )

    def handle(self, *args, **kwargs):
        asyncio.run(self.async_handle(*args, **kwargs))

    async def async_handle(self, *args, **kwargs):
        start_total_time = time.time()
        try:
            self.stdout.write(self.style.SUCCESS("Запуск получения данных..."))
            start_time = time.time()
            await fetch_data()
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Данные получены за {time_used:.2f} сек")
            )

            self.stdout.write(self.style.SUCCESS("Запуск обработки индикаторов..."))
            start_time = time.time()
            await process_all_indicators()
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Индикаторы рассчитаны за {time_used:.2f} сек")
            )

            # self.stdout.write(self.style.SUCCESS("Анализ новостей..."))
            # start_time = time.time()
            # await gather_and_analyze_news("cryptocurrency OR BTC OR ETH OR crypto market")
            # time_used = time.time() - start_time
            # self.stdout.write(self.style.SUCCESS(f"Новости проанализированы за {time_used:.2f} сек"))

            self.stdout.write(self.style.SUCCESS("Расчёт предсказаний..."))
            start_time = time.time()
            await run_short_predictions()  # Асинхронный вызов
            time_used = time.time() - start_time
            self.stdout.write(
                self.style.SUCCESS(f"Предсказания готовы за {time_used:.2f} сек")
            )

            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.SUCCESS(f"Общее время выполнения: {total_time:.2f} сек")
            )

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.WARNING(f"Прервано через {total_time:.2f} сек")
            )
