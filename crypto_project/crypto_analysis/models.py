from django.db import models


class ShortTermCryptoPrediction(models.Model):
    cryptocurrency_pair = models.CharField(max_length=50, db_index=True)
    prediction_date = models.DateField(db_index=True)
    predicted_price_change = models.FloatField()
    predicted_close = models.FloatField()
    model_type = models.CharField(max_length=50)
    confidence_level = models.FloatField(default=1.0)

    class Meta:
        unique_together = ("cryptocurrency_pair", "prediction_date", "model_type")

    def __str__(self):
        return f"{self.cryptocurrency_pair} | {self.prediction_date} | Predicted Change: {self.predicted_price_change:.2f} | Model: {self.model_type} | Confidence: {self.confidence_level:.2f}"


class LongTermCryptoPrediction(models.Model):
    cryptocurrency_pair = models.CharField(max_length=50, db_index=True)
    prediction_date = models.DateField(db_index=True)
    predicted_price_change = models.FloatField()
    predicted_close = models.FloatField()
    model_type = models.CharField(max_length=50)
    confidence_level = models.FloatField(default=1.0)

    class Meta:
        unique_together = ("cryptocurrency_pair", "prediction_date", "model_type")

    def __str__(self):
        return f"{self.cryptocurrency_pair} | {self.prediction_date} | Predicted Change: {self.predicted_price_change:.2f} | Model: {self.model_type} | Confidence: {self.confidence_level:.2f}"


class MarketData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()
    exchange = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        unique_together = ("cryptocurrency", "date", "exchange")

    def __str__(self):
        return f"{self.cryptocurrency} | {self.date} | Close: {self.close_price:.2f}"


class IndicatorData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    price_change_1d = models.FloatField(null=True, blank=True)
    price_change_7d = models.FloatField(null=True, blank=True)
    price_change_14d = models.FloatField(null=True, blank=True)
    price_change_30d = models.FloatField(null=True, blank=True)
    sma_7 = models.FloatField(null=True, blank=True)
    sma_14 = models.FloatField(null=True, blank=True)
    sma_30 = models.FloatField(null=True, blank=True)
    sma_50 = models.FloatField(null=True, blank=True)
    sma_200 = models.FloatField(null=True, blank=True)
    volatility_7d = models.FloatField(null=True, blank=True)
    volatility_14d = models.FloatField(null=True, blank=True)
    volatility_30d = models.FloatField(null=True, blank=True)
    volatility_60d = models.FloatField(null=True, blank=True)
    volatility_180d = models.FloatField(null=True, blank=True)
    rsi_7d = models.FloatField(null=True, blank=True)
    rsi_14d = models.FloatField(null=True, blank=True)
    rsi_30d = models.FloatField(null=True, blank=True)
    rsi_90d = models.FloatField(null=True, blank=True)
    cci_7d = models.FloatField(null=True, blank=True)
    cci_14d = models.FloatField(null=True, blank=True)
    cci_30d = models.FloatField(null=True, blank=True)
    atr_14d = models.FloatField(null=True, blank=True)
    atr_30d = models.FloatField(null=True, blank=True)
    atr_60d = models.FloatField(null=True, blank=True)
    bollinger_bands_14d_upper = models.FloatField(null=True, blank=True)
    bollinger_bands_14d_lower = models.FloatField(null=True, blank=True)
    bollinger_bands_30d_upper = models.FloatField(null=True, blank=True)
    bollinger_bands_30d_lower = models.FloatField(null=True, blank=True)
    macd_12_26 = models.FloatField(null=True, blank=True)
    macd_signal_9 = models.FloatField(null=True, blank=True)
    stochastic_oscillator_7d = models.FloatField(null=True, blank=True)
    stochastic_oscillator_14d = models.FloatField(null=True, blank=True)
    stochastic_oscillator_30d = models.FloatField(null=True, blank=True)
    lag_macd_12 = models.FloatField(null=True, blank=True)
    lag_macd_26 = models.FloatField(null=True, blank=True)
    lag_macd_9 = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('cryptocurrency', 'date')
