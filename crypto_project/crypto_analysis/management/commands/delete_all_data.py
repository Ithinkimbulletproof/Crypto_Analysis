from django.core.management.base import BaseCommand
from crypto_analysis.models import (
    IndicatorData,
    MarketData,
    NewsArticle,
    SentimentData,
    KeyEntity,
)


class Command(BaseCommand):
    help = "Удаляет все данные из указанных моделей"

    def handle(self, *args, **kwargs):
        IndicatorData.objects.all().delete()
        MarketData.objects.all().delete()
        # NewsArticle.objects.all().delete()
        SentimentData.objects.all().delete()
        KeyEntity.objects.all().delete()
        self.stdout.write(self.style.SUCCESS("Все данные успешно удалены"))
