import unittest
import pandas as pd
import numpy as np
from crypto_analysis.services.data_processor import (
    calculate_indicators,
    compute_average_data,
    prepare_data_for_analysis,
    process_data,
    get_features_and_labels,
    scale_and_resample_data,
    main,
)


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_data = [
            {
                "date": "2023-01-01",
                "open_price": 100,
                "high_price": 110,
                "low_price": 90,
                "close_price": 105,
                "volume": 1000,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-02",
                "open_price": 105,
                "high_price": 115,
                "low_price": 95,
                "close_price": 110,
                "volume": 1200,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-03",
                "open_price": 110,
                "high_price": 120,
                "low_price": 100,
                "close_price": 115,
                "volume": 1300,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-04",
                "open_price": 115,
                "high_price": 125,
                "low_price": 105,
                "close_price": 120,
                "volume": 1400,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-05",
                "open_price": 120,
                "high_price": 130,
                "low_price": 110,
                "close_price": 125,
                "volume": 1500,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-06",
                "open_price": 125,
                "high_price": 135,
                "low_price": 115,
                "close_price": 130,
                "volume": 1600,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-07",
                "open_price": 130,
                "high_price": 140,
                "low_price": 120,
                "close_price": 135,
                "volume": 1700,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-08",
                "open_price": 135,
                "high_price": 145,
                "low_price": 125,
                "close_price": 140,
                "volume": 1800,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-09",
                "open_price": 140,
                "high_price": 150,
                "low_price": 130,
                "close_price": 145,
                "volume": 1900,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-10",
                "open_price": 145,
                "high_price": 155,
                "low_price": 135,
                "close_price": 150,
                "volume": 2000,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-11",
                "open_price": 150,
                "high_price": 160,
                "low_price": 140,
                "close_price": 155,
                "volume": 2100,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-12",
                "open_price": 155,
                "high_price": 165,
                "low_price": 145,
                "close_price": 160,
                "volume": 2200,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-13",
                "open_price": 160,
                "high_price": 170,
                "low_price": 150,
                "close_price": 165,
                "volume": 2300,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-14",
                "open_price": 165,
                "high_price": 175,
                "low_price": 155,
                "close_price": 170,
                "volume": 2400,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-15",
                "open_price": 170,
                "high_price": 180,
                "low_price": 160,
                "close_price": 175,
                "volume": 2500,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-16",
                "open_price": 175,
                "high_price": 185,
                "low_price": 165,
                "close_price": 180,
                "volume": 2600,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-17",
                "open_price": 180,
                "high_price": 190,
                "low_price": 170,
                "close_price": 185,
                "volume": 2700,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-18",
                "open_price": 185,
                "high_price": 195,
                "low_price": 175,
                "close_price": 190,
                "volume": 2800,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-19",
                "open_price": 190,
                "high_price": 200,
                "low_price": 180,
                "close_price": 195,
                "volume": 2900,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-20",
                "open_price": 195,
                "high_price": 205,
                "low_price": 185,
                "close_price": 200,
                "volume": 3000,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-21",
                "open_price": 200,
                "high_price": 210,
                "low_price": 190,
                "close_price": 195,
                "volume": 3100,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-22",
                "open_price": 195,
                "high_price": 205,
                "low_price": 185,
                "close_price": 180,
                "volume": 3200,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-23",
                "open_price": 180,
                "high_price": 190,
                "low_price": 170,
                "close_price": 175,
                "volume": 3300,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-24",
                "open_price": 175,
                "high_price": 185,
                "low_price": 165,
                "close_price": 180,
                "volume": 3400,
                "cryptocurrency": "BTC",
            },
            {
                "date": "2023-01-25",
                "open_price": 180,
                "high_price": 190,
                "low_price": 170,
                "close_price": 185,
                "volume": 3500,
                "cryptocurrency": "BTC",
            },
        ]

    def test_calculate_indicators(self):
        df = pd.DataFrame(self.sample_data)
        result_df = calculate_indicators(df)
        self.assertIsNotNone(result_df)
        self.assertIn("SMA_14", result_df.columns)
        self.assertIn("RSI_14", result_df.columns)
        self.assertIn("CCI_14", result_df.columns)
        self.assertIn("volatility", result_df.columns)
        self.assertIn("volume_change", result_df.columns)
        self.assertIn("price_change", result_df.columns)
        self.assertIn("MACD", result_df.columns)
        self.assertIn("Signal_Line", result_df.columns)
        self.assertIn("momentum", result_df.columns)
        self.assertIn("day_of_week", result_df.columns)
        self.assertIn("hour_of_day", result_df.columns)

    def test_compute_average_data(self):
        result_df = compute_average_data(self.sample_data)
        self.assertIsNotNone(result_df)
        self.assertIn("SMA_14", result_df.columns)
        self.assertIn("RSI_14", result_df.columns)
        self.assertIn("CCI_14", result_df.columns)

    def test_prepare_data_for_analysis(self):
        result_df = prepare_data_for_analysis(self.sample_data)
        self.assertIsNotNone(result_df)
        self.assertIn("SMA_14", result_df.columns)
        self.assertIn("RSI_14", result_df.columns)
        self.assertIn("CCI_14", result_df.columns)

    def test_process_data(self):
        df = pd.DataFrame(self.sample_data)
        result_df = process_data(df)
        self.assertIsNotNone(result_df)
        self.assertIn("SMA_14", result_df.columns)
        self.assertIn("RSI_14", result_df.columns)
        self.assertIn("CCI_14", result_df.columns)

    def test_get_features_and_labels(self):
        df = pd.DataFrame(self.sample_data)
        df = calculate_indicators(df)
        X, y = get_features_and_labels(df)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertGreater(len(np.unique(y)), 1)

    def test_scale_and_resample_data(self):
        df = pd.DataFrame(self.sample_data)
        df = calculate_indicators(df)
        X, y = get_features_and_labels(df)

        unique, counts = np.unique(y, return_counts=True)
        print(f"Распределение классов: {dict(zip(unique, counts))}")

        X_resampled, y_resampled = scale_and_resample_data(X, y)
        self.assertIsNotNone(X_resampled)
        self.assertIsNotNone(y_resampled)
        self.assertEqual(len(X_resampled), len(y_resampled))

    def test_main(self):
        X_resampled, y_resampled = main(self.sample_data)
        self.assertIsNotNone(X_resampled)
        self.assertIsNotNone(y_resampled)


if __name__ == "__main__":
    unittest.main()
