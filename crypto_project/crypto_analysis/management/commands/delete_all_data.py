from django.core.management.base import BaseCommand
from crypto_analysis.models import (
    IndicatorData,
    ShortTermCryptoPrediction,
    MarketData,
    NewsArticle,
)


class Command(BaseCommand):
    help = "Удаляет все данные из указанных моделей"

    def handle(self, *args, **kwargs):
        IndicatorData.objects.all().delete()
        ShortTermCryptoPrediction.objects.all().delete()
        MarketData.objects.all().delete()
        NewsArticle.objects.all().delete()
        self.stdout.write(self.style.SUCCESS("Все данные успешно удалены"))
