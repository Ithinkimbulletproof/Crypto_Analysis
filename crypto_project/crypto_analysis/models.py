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


class PreprocessedData(models.Model):
    date = models.DateTimeField()
    close_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    cryptocurrency = models.CharField(max_length=100)
    period = models.CharField(max_length=50)

    price_change_24h = models.FloatField(null=True, blank=True)
    SMA_30 = models.FloatField(null=True, blank=True)
    volatility_30 = models.FloatField(null=True, blank=True)
    SMA_90 = models.FloatField(null=True, blank=True)
    volatility_90 = models.FloatField(null=True, blank=True)
    SMA_180 = models.FloatField(null=True, blank=True)
    volatility_180 = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.cryptocurrency} - {self.date}"


class TechAnalysed(models.Model):
    date = models.DateTimeField()
    cryptocurrency = models.CharField(max_length=100)
    period = models.CharField(max_length=50)
    close_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()

    price_change_24h = models.FloatField(null=True, blank=True)
    SMA_30 = models.FloatField(null=True, blank=True)
    volatility_30 = models.FloatField(null=True, blank=True)
    SMA_90 = models.FloatField(null=True, blank=True)
    volatility_90 = models.FloatField(null=True, blank=True)
    SMA_180 = models.FloatField(null=True, blank=True)
    volatility_180 = models.FloatField(null=True, blank=True)

    predicted_signal = models.IntegerField(null=True, blank=True)
    target = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.cryptocurrency} - {self.date}"
