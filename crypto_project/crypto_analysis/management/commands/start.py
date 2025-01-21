import os
import ccxt
import logging
from datetime import datetime
from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from crypto_analysis.services.data_fetcher import (
    process_exchange_for_all_symbols,
    get_default_start_date,
)
from crypto_analysis.services.data_preprocessing import process_all_indicators
from crypto_analysis.services.machine_learning import process_machine_learning

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Запускает полный пайплайн обработки данных, включая получение, предобработку, технический анализ, машинное обучение и импорт предсказаний."

    def add_arguments(self, parser):
        parser.add_argument(
            "--exchanges",
            type=str,
            help="Список бирж через запятую, например: binance,kraken,okx",
        )

    def handle(self, *args, **kwargs):
        symbols = os.getenv("CRYPTOPAIRS")
        if symbols:
            symbols = symbols.split(",")
        else:
            raise ValueError(
                "Необходимо указать список криптовалют в .env файле через переменную CRYPTOPAIRS."
            )

        exchanges = kwargs["exchanges"]
        if exchanges:
            exchanges = exchanges.split(",")
        else:
            exchanges = []

        try:
            self.stdout.write(self.style.SUCCESS("Шаг 1: Получение данных..."))
            logger.info(f"[{datetime.now()}] Начинаем получение данных для: {symbols}")

            for exchange_id in exchanges:
                try:
                    exchange = getattr(ccxt, exchange_id)()
                    logger.info(f"[{datetime.now()}] Обработка биржи {exchange_id}")
                    process_exchange_for_all_symbols(
                        exchange, symbols, get_default_start_date(), "1h"
                    )
                except Exception as e:
                    logger.error(f"[{datetime.now()}] Ошибка при обработке биржи {exchange_id}: {str(e)}")

            self.stdout.write(self.style.SUCCESS("Получение данных завершено успешно."))

            self.stdout.write(self.style.SUCCESS("Шаг 2: Предобработка данных..."))
            process_all_indicators(symbols)
            self.stdout.write(self.style.SUCCESS("Предобработка данных завершена успешно."))

            self.stdout.write(self.style.SUCCESS("Шаг 3: Запуск моделей машинного обучения..."))
            process_machine_learning()
            self.stdout.write(self.style.SUCCESS("Модели машинного обучения обучены и оценены успешно."))

        except Exception as e:
            logger.error(f"[{datetime.now()}] Ошибка во время выполнения полного пайплайна: {str(e)}")
            self.stderr.write(
                self.style.ERROR(f"[{datetime.now()}] Ошибка во время выполнения полного пайплайна: {str(e)}")
            )
