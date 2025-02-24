import time
import asyncio
import logging
from django.core.management.base import BaseCommand
from crypto_analysis.fetching.data_fetcher import fetch_data
from crypto_analysis.fetching.news_parser import run_full_import
from crypto_analysis.preprocess.data_preprocessing import (
    preprocess_data as indicator_preprocess,
)
from crypto_analysis.preprocess.sentiment_analysis import analyze_sentiment
from crypto_analysis.preprocess.data_aggregator import save_csv_files_by_currency
from crypto_analysis.ml_models.train_update_models import train_and_save_models

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Получает данные с бирж, рассчитывает индикаторы, анализирует новости и сохраняет данные для каждой криптовалюты в отдельный CSV"

    def handle(self, *args, **kwargs):
        asyncio.run(self.async_handle(*args, **kwargs))

    async def async_handle(self, *args, **kwargs):
        start_total_time = time.time()
        try:
            self.stdout.write(self.style.SUCCESS("Запуск получения данных..."))
            start_time = time.time()
            await fetch_data()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Данные получены за {time.time() - start_time:.2f} сек"
                )
            )

            self.stdout.write(self.style.SUCCESS("Запуск расчёта индикаторов..."))
            start_time = time.time()
            await asyncio.to_thread(indicator_preprocess)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Индикаторы рассчитаны за {time.time() - start_time:.2f} сек"
                )
            )

            self.stdout.write(self.style.SUCCESS("Запуск парсинга новостей..."))
            start_time = time.time()
            await asyncio.to_thread(run_full_import)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Новости загружены за {time.time() - start_time:.2f} сек"
                )
            )

            self.stdout.write(self.style.SUCCESS("Запуск анализа настроений..."))
            start_time = time.time()
            await asyncio.to_thread(analyze_sentiment)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Анализ настроений завершён за {time.time() - start_time:.2f} сек"
                )
            )

            self.stdout.write(
                self.style.SUCCESS("Сохранение данных по каждой валюте...")
            )
            start_time = time.time()
            save_path = await asyncio.to_thread(save_csv_files_by_currency)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Файлы сохранены в {save_path} за {time.time() - start_time:.2f} сек"
                )
            )

            self.stdout.write(self.style.SUCCESS("Запуск машинного обучения..."))
            start_time = time.time()
            await asyncio.to_thread(train_and_save_models)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Машинное обучение завершено за {time.time() - start_time:.2f} сек"
                )
            )

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
        finally:
            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.WARNING(f"Процесс завершён через {total_time:.2f} сек")
            )
