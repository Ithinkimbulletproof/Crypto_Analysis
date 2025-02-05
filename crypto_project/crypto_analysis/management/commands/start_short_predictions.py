import time
import asyncio
import logging
import os
from django.core.management.base import BaseCommand
from crypto_analysis.fetching.data_fetcher import fetch_data
from crypto_analysis.fetching.news_parser import run_full_import
from crypto_analysis.preprocess.data_preprocessing import (
    preprocess_data as indicator_preprocess,
)
from crypto_analysis.preprocess.sentiment_analysis import analyze_sentiment
from crypto_analysis.preprocess.data_aggregator import (
    build_unified_dataframe,
    preprocessing_data,
)

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

            self.stdout.write(self.style.SUCCESS("Запуск объединения данных..."))
            start_time = time.time()
            df_unified = await asyncio.to_thread(build_unified_dataframe)
            if df_unified.empty:
                raise ValueError("Объединённый DataFrame пуст!")

            processed_data, features = await asyncio.to_thread(
                preprocessing_data, df_unified
            )

            save_path = os.path.join(os.getcwd(), "data_exports")
            os.makedirs(save_path, exist_ok=True)

            df_unified.to_csv(os.path.join(save_path, "unified_data.csv"), index=False)
            processed_data["df_minmax"].to_csv(
                os.path.join(save_path, "processed_data_minmax.csv"), index=False
            )
            processed_data["df_std"].to_csv(
                os.path.join(save_path, "processed_data_std.csv"), index=False
            )

            self.stdout.write(
                self.style.SUCCESS(
                    f"Объединённые данные сохранены за {time.time() - start_time:.2f} сек"
                )
            )
            self.stdout.write(self.style.SUCCESS(f"Файлы сохранены в {save_path}"))

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Ошибка: {str(e)}"))
        finally:
            total_time = time.time() - start_total_time
            self.stdout.write(
                self.style.WARNING(f"Процесс завершён через {total_time:.2f} сек")
            )
