from django.db import models


class CryptoPrediction(models.Model):
    cryptocurrency_pair = models.CharField(max_length=50)
    prediction_date = models.DateField()
    predicted_price_change = models.FloatField()
    probability_increase = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )
    probability_decrease = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )

    class Meta:
        unique_together = ("cryptocurrency_pair", "prediction_date")

    def __str__(self):
        return f"{self.cryptocurrency_pair} | {self.prediction_date} | Predicted Change: {self.predicted_price_change:.2f}"


class MarketData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()
    exchange = models.CharField(max_length=50, null=True, blank=True)

    sma_14 = models.FloatField(null=True, blank=True)
    rsi_14 = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("cryptocurrency", "date", "exchange")

    def __str__(self):
        return f"{self.cryptocurrency} | {self.date} | Close: {self.close_price:.2f}"
