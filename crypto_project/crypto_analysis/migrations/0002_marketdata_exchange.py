# Generated by Django 5.1.4 on 2025-01-02 18:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("crypto_analysis", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="marketdata",
            name="exchange",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
