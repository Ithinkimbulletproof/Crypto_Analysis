from django.contrib import admin
from .models import MarketData


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
        "predicted_price_change",
    )
    list_filter = ("cryptocurrency", "date")
    search_fields = ("cryptocurrency",)
    date_hierarchy = "date"
    ordering = ("-date",)
