from django.contrib import admin
from crypto_analysis.models import (
    ShortTermCryptoPrediction,
    LongTermCryptoPrediction,
    MarketData,
    IndicatorData,
    NewsArticle,
)


@admin.register(ShortTermCryptoPrediction)
class ShortTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency_pair",
        "prediction_date",
        "predicted_price_change",
        "predicted_close",
        "current_price",
        "model_type",
        "confidence_level",
    )
    list_filter = ("cryptocurrency_pair", "prediction_date", "model_type")
    search_fields = ("cryptocurrency_pair", "model_type")
    ordering = ("-prediction_date",)


@admin.register(LongTermCryptoPrediction)
class LongTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency_pair",
        "prediction_date",
        "predicted_price_change",
        "predicted_close",
        "model_type",
        "confidence_level",
    )
    list_filter = ("cryptocurrency_pair", "prediction_date", "model_type")
    search_fields = ("cryptocurrency_pair", "model_type")
    ordering = ("-prediction_date",)


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency",
        "date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "exchange",
    )
    list_filter = ("cryptocurrency", "exchange", "date")
    search_fields = ("cryptocurrency", "exchange")
    ordering = ("-date",)


@admin.register(IndicatorData)
class IndicatorDataAdmin(admin.ModelAdmin):
    list_display = ("cryptocurrency", "date", "indicator_name", "value")
    list_filter = ("cryptocurrency", "indicator_name", "date")
    search_fields = ("cryptocurrency", "indicator_name")
    ordering = ("-date",)


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "published_at",
        "sentiment",
        "polarity",
        "source",
        "language",
    )
    list_filter = ("sentiment", "language", "published_at")
    search_fields = ("title", "description", "source")
    ordering = ("-published_at",)
