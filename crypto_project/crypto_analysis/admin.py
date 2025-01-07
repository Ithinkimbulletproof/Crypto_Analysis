from django.contrib import admin
from .models import MarketData, Prediction


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
        "sma_14",
        "rsi_14",
    )
    list_filter = ("cryptocurrency", "date")
    search_fields = ("cryptocurrency",)
    date_hierarchy = "date"
    ordering = ("-date",)


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "cryptocurrency",
        "prediction_date",
        "predicted_price_change",
        "probability_increase",
        "probability_decrease",
    )
    list_filter = ("cryptocurrency", "prediction_date")
    search_fields = ("cryptocurrency",)
    date_hierarchy = "prediction_date"
    ordering = ("-prediction_date",)
