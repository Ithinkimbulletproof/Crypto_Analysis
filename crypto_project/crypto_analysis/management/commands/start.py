from django.core.management.base import BaseCommand
from crypto_analysis.services.data_fetcher import fetch_data
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Получает данные с бирж и сохраняет их в базе данных'

    def handle(self, *args, **kwargs):
        try:
            self.stdout.write(self.style.SUCCESS('Запуск получения данных...'))
            fetch_data()
            self.stdout.write(self.style.SUCCESS('Данные успешно получены и сохранены'))
        except Exception as e:
            logger.error(f"Ошибка при получении данных: {str(e)}")
            self.stdout.write(self.style.ERROR(f'Ошибка: {str(e)}'))
