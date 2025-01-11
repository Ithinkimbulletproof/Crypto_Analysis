from django.contrib import admin
from .models import MarketData, CryptoPrediction


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
    )
    list_filter = ("cryptocurrency", "date")
    search_fields = ("cryptocurrency",)
    date_hierarchy = "date"
    ordering = ("-date",)


@admin.register(CryptoPrediction)
class CryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency_pair",
        "prediction_date",
        "predicted_price_change",
        "probability_increase",
        "probability_decrease",
    )
    list_filter = ("cryptocurrency_pair", "prediction_date")
    search_fields = ("cryptocurrency_pair",)
    date_hierarchy = "prediction_date"
    ordering = ("-prediction_date",)
