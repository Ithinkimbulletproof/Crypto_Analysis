from django.contrib import admin
from .models import (
    ShortTermCryptoPrediction,
    LongTermCryptoPrediction,
    MarketData,
    IndicatorData,
)


class ShortTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency_pair",
        "prediction_date",
        "predicted_price_change",
        "predicted_close",
        "model_type",
        "confidence_level",
    )
    list_filter = ("cryptocurrency_pair", "prediction_date", "model_type")
    search_fields = ("cryptocurrency_pair",)
    date_hierarchy = "prediction_date"
    ordering = ("-prediction_date",)


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
    search_fields = ("cryptocurrency_pair",)
    date_hierarchy = "prediction_date"
    ordering = ("-prediction_date",)


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
    list_filter = ("cryptocurrency", "date", "exchange")
    search_fields = ("cryptocurrency", "exchange")
    date_hierarchy = "date"
    ordering = ("-date",)


class IndicatorDataAdmin(admin.ModelAdmin):
    list_display = ("cryptocurrency", "date", "indicator_name", "value")
    list_filter = ("cryptocurrency", "indicator_name")
    search_fields = ("cryptocurrency", "indicator_name")
    date_hierarchy = "date"
    ordering = ("-date",)


admin.site.register(ShortTermCryptoPrediction, ShortTermCryptoPredictionAdmin)
admin.site.register(LongTermCryptoPrediction, LongTermCryptoPredictionAdmin)
admin.site.register(MarketData, MarketDataAdmin)
admin.site.register(IndicatorData, IndicatorDataAdmin)
