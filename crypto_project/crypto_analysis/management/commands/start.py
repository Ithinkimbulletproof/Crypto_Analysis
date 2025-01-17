import os
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from crypto_analysis.services.data_fetcher import fetch_data
from crypto_analysis.services.data_preprocessing import process_and_export_data
from crypto_analysis.services.tech_analysis import process_and_evaluate_data
from crypto_analysis.services.machine_learning import process_machine_learning
from crypto_analysis.services.predictions import (
    import_short_term_predictions,
    import_long_term_predictions,
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Запускает полный пайплайн обработки данных, включая получение, предобработку, технический анализ, машинное обучение и импорт предсказаний."

    def handle(self, *args, **kwargs):
        try:
            self.stdout.write(self.style.SUCCESS("Шаг 1: Получение данных..."))
            fetch_data()
            self.stdout.write(self.style.SUCCESS("Получение данных завершено успешно."))

            self.stdout.write(self.style.SUCCESS("Шаг 2: Предобработка данных..."))
            process_and_export_data()
            self.stdout.write(
                self.style.SUCCESS("Предобработка данных завершена успешно.")
            )

            self.stdout.write(
                self.style.SUCCESS("Шаг 3: Выполнение технического анализа...")
            )
            process_and_evaluate_data()
            self.stdout.write(
                self.style.SUCCESS("Технический анализ завершен успешно.")
            )

            self.stdout.write(
                self.style.SUCCESS("Шаг 4: Запуск моделей машинного обучения...")
            )
            process_machine_learning()
            self.stdout.write(
                self.style.SUCCESS(
                    "Модели машинного обучения обучены и оценены успешно."
                )
            )

            short_term_file_path = os.path.join(
                settings.BASE_DIR, "results", "short_term_predictions.csv"
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Шаг 5: Импорт краткосрочных предсказаний из {short_term_file_path}..."
                )
            )
            import_short_term_predictions(short_term_file_path)
            self.stdout.write(
                self.style.SUCCESS("Краткосрочные предсказания импортированы успешно.")
            )

            long_term_file_path = os.path.join(
                settings.BASE_DIR, "results", "long_term_predictions.csv"
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Шаг 6: Импорт долгосрочных предсказаний из {long_term_file_path}..."
                )
            )
            import_long_term_predictions(long_term_file_path)
            self.stdout.write(
                self.style.SUCCESS("Долгосрочные предсказания импортированы успешно.")
            )

            self.stdout.write(
                self.style.SUCCESS("Полный пайплайн обработки данных завершен успешно.")
            )
        except Exception as e:
            logger.error(f"Ошибка во время выполнения полного пайплайна: {str(e)}")
            self.stderr.write(
                self.style.ERROR(
                    f"Ошибка во время выполнения полного пайплайна: {str(e)}"
                )
            )
