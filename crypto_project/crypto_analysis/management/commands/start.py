from django.core.management.base import BaseCommand
from crypto_analysis.services.data_fetcher import fetch_data
from crypto_analysis.services.predictions import analyze_and_update


class Command(BaseCommand):
    help = "Получение данных, обрабаботка и анализ крипторынка"

    def handle(self, *args, **kwargs):
        fetch_data()
        analyze_and_update()
        self.stdout.write(self.style.SUCCESS("Процесс завершен"))
