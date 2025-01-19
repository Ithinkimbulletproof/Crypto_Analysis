import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from crypto_analysis.models import MarketData
from crypto_analysis.services.data_fetcher import (
    get_last_date,
    fetch_data,
    process_exchange,
    store_data_bulk,
)


@pytest.fixture
def create_test_data():
    MarketData.objects.create(
        cryptocurrency="BTC/USDT",
        date=datetime(2021, 6, 1, tzinfo=timezone.utc),
        open_price=35000.00,
        high_price=36000.00,
        low_price=34000.00,
        close_price=35500.00,
        volume=1500.0,
        exchange="binance",
    )
    MarketData.objects.create(
        cryptocurrency="ETH/USDT",
        date=datetime(2021, 6, 1, tzinfo=timezone.utc),
        open_price=2500.00,
        high_price=2600.00,
        low_price=2400.00,
        close_price=2550.00,
        volume=5000.0,
        exchange="kraken",
    )
    return "binance", "ETH/USDT", "BTC/USDT"


@pytest.mark.django_db
def test_get_last_date(create_test_data):
    exchange = MagicMock()
    exchange.id = "binance"
    symbol = "BTC/USDT"

    last_date = get_last_date(symbol, exchange)

    assert last_date == datetime(2021, 6, 1, tzinfo=timezone.utc)


@pytest.mark.django_db
def test_fetch_data(create_test_data):
    exchanges = [MagicMock(), MagicMock()]
    exchanges[0].id = "binance"
    exchanges[1].id = "kraken"
    exchanges[0].fetch_ohlcv = MagicMock(
        return_value=[
            [1622558400000, 35000, 36000, 34000, 35500, 1500],
        ]
    )
    exchanges[1].fetch_ohlcv = MagicMock(
        return_value=[
            [1622558400000, 2500, 2600, 2400, 2550, 5000],
        ]
    )
    symbols = ["BTC/USDT", "ETH/USDT"]

    fetch_data()

    assert MarketData.objects.filter(cryptocurrency="BTC/USDT").exists()
    assert MarketData.objects.filter(cryptocurrency="ETH/USDT").exists()


@pytest.mark.django_db
def test_process_exchange(create_test_data):
    exchange = MagicMock()
    exchange.id = "binance"
    symbols = ["BTC/USDT", "ETH/USDT"]
    default_start_date = 1609459200000
    timeframe = "1h"

    exchange.load_markets = MagicMock(return_value=None)
    exchange.fetch_ohlcv = MagicMock(
        return_value=[
            [1622558400000, 35000, 36000, 34000, 35500, 1500],
        ]
    )

    process_exchange(exchange, symbols, default_start_date, timeframe)

    assert MarketData.objects.filter(cryptocurrency="BTC/USDT").exists()


@pytest.mark.django_db
def test_store_data_bulk(create_test_data):
    all_data = [
        ("BTC/USDT", "binance", 1622558400000, 35000, 36000, 34000, 35500, 1500),
        ("ETH/USDT", "kraken", 1622558400000, 2500, 2600, 2400, 2550, 5000),
    ]

    store_data_bulk(all_data)

    assert MarketData.objects.filter(cryptocurrency="BTC/USDT").exists()
    assert MarketData.objects.filter(cryptocurrency="ETH/USDT").exists()
