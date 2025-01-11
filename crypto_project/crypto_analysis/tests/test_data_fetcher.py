from django.test import TestCase
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone as dt_timezone
from django.utils import timezone as django_timezone
import logging
from crypto_analysis.models import MarketData
from crypto_analysis.services.data_fetcher import (
    get_default_start_date,
    fetch_data,
    process_exchange,
    fetch_and_store_data,
    store_data,
    get_last_date,
    get_since,
)

logger = logging.getLogger(__name__)


class TestDataFetcher(TestCase):
    def setUp(self):
        self.exchange_mock = MagicMock()
        self.exchange_mock.id = "binance"
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.default_start_date = 1609459200000

    @patch("crypto_analysis.services.data_fetcher.datetime")
    def test_get_default_start_date(self, datetime_mock):
        local_time = datetime(2022, 1, 1, 0, 0, 0, tzinfo=dt_timezone.utc)
        utc_time = local_time.astimezone(dt_timezone.utc)
        datetime_mock.now.return_value = utc_time
        start_date = get_default_start_date()
        expected_start_date = 1609459200000
        self.assertEqual(start_date, expected_start_date)

    @patch(
        "crypto_analysis.services.data_fetcher.get_default_start_date",
        return_value=1609459200000,
    )
    @patch("crypto_analysis.services.data_fetcher.process_exchange")
    def test_fetch_data_calls_process_exchange(
        self, process_exchange_mock, get_default_start_date_mock
    ):
        fetch_data()
        process_exchange_mock.assert_called_once()
        logger.info("process_exchange был вызван один раз")

    @patch("crypto_analysis.services.data_fetcher.get_last_date")
    @patch("crypto_analysis.services.data_fetcher.fetch_and_store_data")
    def test_process_exchange_calls_fetch_and_store_data(
        self, fetch_and_store_data_mock, get_last_date_mock
    ):
        get_last_date_mock.return_value = None
        symbols = [self.symbol]
        self.exchange_mock.markets = {self.symbol: {}}
        process_exchange(self.exchange_mock, symbols, self.default_start_date)
        fetch_and_store_data_mock.assert_called_once_with(
            self.exchange_mock, self.symbol, self.default_start_date
        )
        logger.info("fetch_and_store_data был вызван один раз")

    @patch("crypto_analysis.services.data_fetcher.timezone.now")
    def test_get_last_date_returns_correct_date(self, timezone_now_mock):
        now = django_timezone.make_aware(datetime(2021, 1, 1))
        timezone_now_mock.return_value = now
        MarketData.objects.create(
            cryptocurrency=self.symbol,
            date=now,
            exchange="binance",
            open_price=10000,
            high_price=10500,
            low_price=9500,
            close_price=10200,
            volume=100,
        )
        last_date = get_last_date(self.symbol, self.exchange_mock)
        self.assertEqual(last_date, now)
        logger.info(f"Последняя дата для {self.symbol}: {last_date}")

    def test_get_since_returns_correct_timestamp(self):
        last_date = django_timezone.make_aware(datetime(2021, 1, 1))
        since = get_since(last_date, self.default_start_date)
        self.assertEqual(since, int(last_date.timestamp() * 1000))
        logger.info(f"Получен timestamp для since: {since}")

    @patch("crypto_analysis.services.data_fetcher.store_data")
    def test_fetch_and_store_data_calls_store_data(self, store_data_mock):
        symbol = self.symbol
        since = self.default_start_date
        mock_data = [
            [[1609459200000, 10000, 10500, 9500, 10200, 100]],
            [[1609462800000, 10200, 10700, 9700, 10300, 150]],
            [],
        ]

        def side_effect(*args, **kwargs):
            nonlocal mock_data
            return mock_data.pop(0)

        self.exchange_mock.fetch_ohlcv.side_effect = side_effect
        fetch_and_store_data(self.exchange_mock, symbol, since)
        self.assertEqual(store_data_mock.call_count, 2)
        logger.info(f"store_data был вызван {store_data_mock.call_count} раз")

    @patch(
        "crypto_analysis.services.data_fetcher.get_default_start_date",
        return_value=1609459200000,
    )
    def test_store_data_creates_market_data(self, get_default_start_date_mock):
        symbol = self.symbol
        record = [1609459200000, 10000, 10500, 9500, 10200, 100]
        naive_date = datetime.utcfromtimestamp(record[0] / 1000)
        aware_date = django_timezone.make_aware(naive_date, timezone=dt_timezone.utc)

        store_data(symbol, self.exchange_mock, record)

        market_data = MarketData.objects.first()
        self.assertIsNotNone(market_data, "MarketData не создан")
        self.assertEqual(market_data.cryptocurrency, symbol, "Неверный cryptocurrency")
        self.assertEqual(market_data.date, aware_date, "Неверный date")
        self.assertEqual(market_data.open_price, record[1], "Неверный open_price")
        self.assertEqual(market_data.high_price, record[2], "Неверный high_price")
        self.assertEqual(market_data.low_price, record[3], "Неверный low_price")
        self.assertEqual(market_data.close_price, record[4], "Неверный close_price")
        self.assertEqual(market_data.volume, record[5], "Неверный volume")
