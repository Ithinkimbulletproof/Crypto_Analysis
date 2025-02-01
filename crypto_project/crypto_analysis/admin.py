from django.contrib import admin
from .models import MarketData, NewsArticle, IndicatorData, SentimentData, KeyEntity


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ("cryptocurrency", "date", "open_price", "close_price", "volume")
    list_filter = ("cryptocurrency", "date")
    search_fields = ("cryptocurrency",)


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ("title", "source", "published_at", "language")
    list_filter = ("source", "language")
    search_fields = ("title", "description")


@admin.register(IndicatorData)
class IndicatorDataAdmin(admin.ModelAdmin):
    list_display = ("cryptocurrency", "indicator_name", "date", "value")
    list_filter = ("cryptocurrency", "indicator_name")
    search_fields = ("cryptocurrency", "indicator_name")


@admin.register(SentimentData)
class SentimentDataAdmin(admin.ModelAdmin):
    list_display = (
        "article",
        "vader_compound",
        "bert_positive",
        "combined_score",
        "analyzed_at",
    )
    list_filter = ("combined_score", "bert_positive")
    search_fields = ("article__title",)


@admin.register(KeyEntity)
class KeyEntityAdmin(admin.ModelAdmin):
    list_display = ("article", "entity_type", "text", "count")
    list_filter = ("entity_type",)
    search_fields = ("text", "article__title")
