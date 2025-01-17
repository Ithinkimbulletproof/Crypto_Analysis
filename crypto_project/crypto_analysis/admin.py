from django.contrib import admin
from .models import MarketData, ShortTermCryptoPrediction, LongTermCryptoPrediction


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
    date_hierarchy = "date"
    ordering = ("-date",)


@admin.register(ShortTermCryptoPrediction)
class ShortTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency_pair",
        "prediction_date",
        "predicted_price_change",
        "predicted_close",
        "model_type",
        "confidence_level",
    )
    list_filter = ("cryptocurrency_pair", "model_type", "prediction_date")
    search_fields = ("cryptocurrency_pair", "model_type")
    date_hierarchy = "prediction_date"
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
    list_filter = ("cryptocurrency_pair", "model_type", "prediction_date")
    search_fields = ("cryptocurrency_pair", "model_type")
    date_hierarchy = "prediction_date"
    ordering = ("-prediction_date",)
