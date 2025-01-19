import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from crypto_analysis.models import MarketData, PreprocessedData
from crypto_analysis.services.data_preprocessing import (
    fetch_data_from_database,
    preprocess_data,
    split_data_by_period,
    save_to_database,
    process_and_export_data,
)


@pytest.mark.django_db
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
    return "BTC/USDT"


@pytest.mark.django_db
def test_preprocess_data(create_test_data):
    crypto = "BTC/USDT"
    data = fetch_data_from_database(crypto)

    if len(data) < 2:
        data.append(
            (datetime(2021, 6, 2, tzinfo=timezone.utc), 36000.00, 37000.00, 35000.00, "BTC/USDT")
        )

    if len(data) < 24:
        for i in range(len(data), 24):
            data.append(
                (datetime(2021, 6, 3, tzinfo=timezone.utc) + timedelta(days=i), 36000.00, 37000.00, 35000.00, "BTC/USDT")
            )

    df = preprocess_data(data)

    assert not df.empty or len(df) == 1

    if not df.empty:
        if len(df) > 1:
            assert "price_change_24h" in df.columns
            assert "volatility_30" in df.columns

    assert len(df) > 0 or len(df) == 1


@pytest.mark.django_db
@patch("crypto_analysis.services.data_preprocessing.save_to_database")
def test_process_and_export_data(mock_save_to_database, create_test_data):
    with patch("crypto_analysis.services.data_preprocessing.fetch_data_from_database", return_value=[
        (datetime(2021, 6, 1, tzinfo=timezone.utc), 35000.00, 36000.00, 34000.00, "BTC/USDT")
    ]):
        process_and_export_data(volatility_window=30)

    assert mock_save_to_database.called
    assert mock_save_to_database.call_count > 0


@pytest.mark.django_db
def test_split_data_by_period(create_test_data):
    crypto = "BTC/USDT"
    data = fetch_data_from_database(crypto)
    df = preprocess_data(data)
    periods = [30, 90, 180]

    split_data = split_data_by_period(df, periods)

    for period in periods:
        assert period in split_data
        assert not split_data[period].empty


@pytest.mark.django_db
def test_save_to_database(create_test_data):
    crypto = "BTC/USDT"
    data = fetch_data_from_database(crypto)
    df = preprocess_data(data)
    split_data = split_data_by_period(df, [30])

    save_to_database(split_data[30], crypto, 30)

    assert PreprocessedData.objects.filter(cryptocurrency=crypto).exists()


@pytest.mark.django_db
def test_save_to_database_empty_df(create_test_data):
    crypto = "BTC/USDT"
    df = pd.DataFrame(columns=["date", "close", "high", "low", "cryptocurrency"])

    save_to_database(df, crypto, 30)

    assert PreprocessedData.objects.count() == 0


@pytest.mark.django_db
def test_preprocess_data_empty(create_test_data):
    data = []
    df = preprocess_data(data)

    assert df.empty
    assert "date" in df.columns
    assert "close" in df.columns
