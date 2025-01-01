from django.db import models


class MarketData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()

    sma_14 = models.FloatField(null=True, blank=True)
    rsi_14 = models.FloatField(null=True, blank=True)
    predicted_price_change = models.FloatField(null=True, blank=True)
    probability_increase = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    probability_decrease = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)

    class Meta:
        unique_together = ("cryptocurrency", "date")

    def __str__(self):
        return f"{self.cryptocurrency} | {self.date} | Close: {self.close_price:.2f}"
