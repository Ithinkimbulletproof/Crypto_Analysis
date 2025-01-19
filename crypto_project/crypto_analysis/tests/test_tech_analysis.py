import pytest
import pandas as pd
from datetime import datetime
from crypto_analysis.models import PreprocessedData, TechAnalysed
from crypto_analysis.services.tech_analysis import (
    fetch_data_from_db,
    split_data_by_period,
    enhance_data_processing,
    generate_target_variable,
    apply_technical_analysis,
    save_to_db,
    evaluate_model_performance,
    process_and_evaluate_data,
    log_overall_stats
)

import pytest

@pytest.fixture(scope='function')
def setup_db(db):
    assert PreprocessedData.objects.exists(), "No data found in PreprocessedData"

    yield

    TechAnalysed.objects.all().delete()

@pytest.mark.django_db
def test_fetch_data_from_db(setup_db):
    cryptocurrency = "BTC/USDT"
    period = 30
    df = fetch_data_from_db(cryptocurrency, period)

    assert df is not None
    assert df.shape[0] > 0
    assert "close_price" in df.columns


def test_split_data_by_period():
    data = {
        'date': pd.to_datetime(['2025-01-19', '2025-01-18']),
        'close': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'RSI': [50, 55],
    }
    df = pd.DataFrame(data)
    split_data = split_data_by_period(df, periods=[30, 90])

    assert len(split_data[30]) == 2
    assert len(split_data[90]) == 0


def test_enhance_data_processing():
    data = {
        'date': pd.to_datetime(['2025-01-19', '2025-01-18']),
        'close': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'RSI': [50, 55],
    }
    df = pd.DataFrame(data)
    df_enhanced = enhance_data_processing(df)

    assert df_enhanced is not None
    assert "close_lag_1" in df_enhanced.columns
    assert "RSI_lag_1" in df_enhanced.columns


def test_generate_target_variable():
    data = {
        'date': pd.to_datetime(['2025-01-19', '2025-01-18']),
        'close': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'RSI': [50, 55],
    }
    df = pd.DataFrame(data)
    df_target = generate_target_variable(df, period=1)

    assert "target" in df_target.columns
    assert df_target["target"].iloc[0] == 0


def test_apply_technical_analysis():
    data = {
        'date': pd.to_datetime(['2025-01-19', '2025-01-18']),
        'close': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'RSI': [50, 55],
    }
    df = pd.DataFrame(data)
    df_tech = apply_technical_analysis(df)

    assert "SMA_50" in df_tech.columns
    assert "RSI" in df_tech.columns
    assert "predicted_signal" in df_tech.columns


def test_save_to_db(setup_db):
    data = PreprocessedData.objects.all().values('date', 'close_price', 'high_price', 'low_price', 'cryptocurrency')
    df = pd.DataFrame(data)

    cryptocurrency = "BTC/USDT"
    period = 30
    save_to_db(df, cryptocurrency, period)

    assert TechAnalysed.objects.count() > 0


def test_evaluate_model_performance():
    data = {
        'date': pd.to_datetime(['2025-01-19', '2025-01-18']),
        'close_price': [100, 101],
        'high_price': [102, 103],
        'low_price': [99, 100],
        'RSI': [50, 55],
        'predicted_signal': [1, 0],
        'target': [1, 0],
    }
    df = pd.DataFrame(data)

    try:
        evaluate_model_performance(df)
    except Exception as e:
        assert str(e) is None


def test_process_and_evaluate_data(setup_db):
    process_and_evaluate_data()

    assert TechAnalysed.objects.count() > 0
    assert log_overall_stats is not None
