from django.shortcuts import render
from crypto_analysis.models import MarketData


def dashboard(request):
    market_data = MarketData.objects.all()
    return render(request, "dashboard.html", {"market_data": market_data})
