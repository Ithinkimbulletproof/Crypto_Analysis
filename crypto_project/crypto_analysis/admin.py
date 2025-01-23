from django.contrib import admin
from .models import ShortTermCryptoPrediction, LongTermCryptoPrediction, MarketData, IndicatorData

class ShortTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'cryptocurrency_pair', 'prediction_date', 'predicted_price_change',
        'predicted_close', 'model_type', 'confidence_level'
    )
    list_filter = ('model_type', 'prediction_date')
    search_fields = ('cryptocurrency_pair', 'model_type')
    ordering = ('-prediction_date',)

class LongTermCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'cryptocurrency_pair', 'prediction_date', 'predicted_price_change',
        'predicted_close', 'model_type', 'confidence_level'
    )
    list_filter = ('model_type', 'prediction_date')
    search_fields = ('cryptocurrency_pair', 'model_type')
    ordering = ('-prediction_date',)

class MarketDataAdmin(admin.ModelAdmin):
    list_display = (
        'cryptocurrency', 'date', 'open_price', 'high_price', 'low_price',
        'close_price', 'volume', 'exchange'
    )
    list_filter = ('cryptocurrency', 'exchange')
    search_fields = ('cryptocurrency',)
    ordering = ('-date',)

class IndicatorDataAdmin(admin.ModelAdmin):
    list_display = (
        'cryptocurrency', 'date', 'price_change_1d', 'price_change_7d',
        'price_change_14d', 'price_change_30d', 'sma_7', 'sma_14', 'sma_30',
        'sma_50', 'sma_200', 'volatility_7d', 'volatility_14d', 'volatility_30d',
        'volatility_60d', 'volatility_180d', 'rsi_7d', 'rsi_14d', 'rsi_30d',
        'rsi_90d', 'cci_7d', 'cci_14d', 'cci_30d', 'atr_14d', 'atr_30d',
        'atr_60d', 'bollinger_bands_14d_upper', 'bollinger_bands_14d_lower',
        'bollinger_bands_30d_upper', 'bollinger_bands_30d_lower', 'macd_12_26',
        'macd_signal_9', 'stochastic_oscillator_7d', 'stochastic_oscillator_14d',
        'stochastic_oscillator_30d', 'lag_macd_12', 'lag_macd_26', 'lag_macd_9'
    )
    list_filter = ('cryptocurrency',)
    search_fields = ('cryptocurrency',)
    ordering = ('-date',)

admin.site.register(ShortTermCryptoPrediction, ShortTermCryptoPredictionAdmin)
admin.site.register(LongTermCryptoPrediction, LongTermCryptoPredictionAdmin)
admin.site.register(MarketData, MarketDataAdmin)
admin.site.register(IndicatorData, IndicatorDataAdmin)
