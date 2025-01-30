import time
import asyncio
import logging
from django.core.management.base import BaseCommand
from crypto_analysis.fetching.data_fetcher import fetch_data


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

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.WARNING(f"Прервано через {total_time:.2f} сек")
            )
