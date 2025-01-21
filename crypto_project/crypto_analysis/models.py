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
    indicator_name = models.CharField(max_length=100)
    value = models.FloatField()

    class Meta:
        unique_together = ("cryptocurrency", "date", "indicator_name")
        indexes = [
            models.Index(fields=['cryptocurrency', 'date']),
            models.Index(fields=['cryptocurrency', 'indicator_name']),
            models.Index(fields=['date', 'indicator_name']),
        ]

    def __str__(self):
        return f"{self.cryptocurrency} - {self.indicator_name} - {self.date}"
