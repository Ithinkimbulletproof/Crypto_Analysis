import pytest
import pandas as pd
from unittest.mock import patch
from crypto_analysis.models import PreprocessedData, TechAnalysed
from crypto_analysis.fetching.tech_analysis import (
    fetch_data_from_db,
    split_data_by_period,
    enhance_data_processing,
    generate_target_variable,
    apply_technical_analysis,
    save_to_db,
    evaluate_model_performance,
)


@pytest.fixture
def preprocessed_data():
    return PreprocessedData.objects.create(
        date="2025-01-01",
        close_price=100.0,
        high_price=105.0,
        low_price=95.0,
        cryptocurrency="BTC",
        period="30",
        price_change_24h=2.0,
        SMA_30=99.5,
        volatility_30=1.5,
    )


def test_fetch_data_from_db(preprocessed_data):
    with patch("myapp.fetch_data_from_db") as mock_fetch:
        mock_fetch.return_value = preprocessed_data
        data = fetch_data_from_db("BTC", "30")
        assert data is not None
        assert data["cryptocurrency"] == "BTC"


def test_split_data_by_period():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
            "close": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
        }
    )
    result = split_data_by_period(df, periods=[30, 90])
    assert len(result[30]) == 5
    assert len(result[90]) == 5


def test_enhance_data_processing():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
            "close": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "RSI": [30, 40, 50, 60, 70],
        }
    )
    processed_df = enhance_data_processing(df)
    assert "close_lag_1" in processed_df.columns
    assert "RSI_lag_1" in processed_df.columns


def test_apply_technical_analysis():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
            "close": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
        }
    )
    analyzed_df = apply_technical_analysis(df)
    assert "SMA_50" in analyzed_df.columns
    assert "RSI" in analyzed_df.columns


def test_generate_target_variable():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
            "close": [100, 101, 102, 103, 104],
        }
    )
    target_df = generate_target_variable(df, period=1)
    assert "target" in target_df.columns
    assert target_df["target"].iloc[-1] == 1


def test_save_to_db():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
            "close": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "predicted_signal": [1, -1, 1, -1, 1],
            "target": [1, -1, 1, -1, 1],
        }
    )
    save_to_db(df, "BTC", 30)
    assert TechAnalysed.objects.count() > 0
    assert TechAnalysed.objects.first().cryptocurrency == "BTC"


@pytest.mark.django_db
def test_evaluate_model_performance():
    df = pd.DataFrame(
        {"target": [1, 1, -1, -1, 1], "predicted_signal": [1, 1, -1, 1, 1]}
    )
    evaluate_model_performance(df)
