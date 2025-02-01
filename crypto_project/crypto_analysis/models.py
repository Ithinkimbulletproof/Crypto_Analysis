from django.db import models


class MarketData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()

    class Meta:
        unique_together = ("cryptocurrency", "date")

    def __str__(self):
        return f"{self.cryptocurrency} | {self.date} | Close: {self.close_price:.2f}"


class NewsArticle(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(max_length=2000)
    url = models.URLField(unique=True)
    published_at = models.DateTimeField()
    source = models.CharField(max_length=50)
    language = models.CharField(max_length=10, default="en")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["published_at"]),
            models.Index(fields=["source"]),
        ]
        ordering = ["-published_at"]

    def __str__(self):
        return self.title[:50]


class IndicatorData(models.Model):
    cryptocurrency = models.CharField(max_length=50)
    date = models.DateTimeField()
    indicator_name = models.CharField(max_length=100)
    value = models.FloatField()

    class Meta:
        unique_together = ("cryptocurrency", "date", "indicator_name")
        indexes = [
            models.Index(fields=["cryptocurrency", "date"]),
            models.Index(fields=["cryptocurrency", "indicator_name"]),
            models.Index(fields=["date", "indicator_name"]),
        ]

    def __str__(self):
        return f"{self.cryptocurrency} - {self.indicator_name} - {self.date}"


class SentimentData(models.Model):
    article = models.OneToOneField(
        NewsArticle, on_delete=models.CASCADE, related_name="sentiment_data"
    )
    vader_compound = models.FloatField()
    bert_positive = models.FloatField()
    emotion_scores = models.JSONField(default=dict)
    topic_scores = models.JSONField(default=dict)
    combined_score = models.FloatField()
    analyzed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["combined_score"]),
            models.Index(fields=["bert_positive"]),
        ]


class KeyEntity(models.Model):
    ENTITY_TYPES = [
        ("ORG", "Organization"),
        ("PRODUCT", "Product"),
        ("GPE", "Location"),
        ("MONEY", "Money"),
    ]

    article = models.ForeignKey(
        NewsArticle, on_delete=models.CASCADE, related_name="entities"
    )
    entity_type = models.CharField(max_length=20, choices=ENTITY_TYPES)
    text = models.CharField(max_length=255)
    count = models.IntegerField(default=1)

    class Meta:
        unique_together = ("article", "entity_type", "text")
        indexes = [
            models.Index(fields=["entity_type"]),
            models.Index(fields=["text"]),
        ]
