import unittest
import pandas as pd
from crypto_analysis.services.data_processor import compute_average_data

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Создаем тестовые данные с достаточным количеством записей
        self.test_data = [
            {
                "date": "2023-12-01 12:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35000,
                "high_price": 36000,
                "low_price": 34000,
                "close_price": 35500,
                "volume": 1000,
            },
            {
                "date": "2023-12-01 13:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35500,
                "high_price": 36500,
                "low_price": 34500,
                "close_price": 36000,
                "volume": 1200,
            },
            {
                "date": "2023-12-01 14:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36000,
                "high_price": 37000,
                "low_price": 35000,
                "close_price": 36500,
                "volume": 1300,
            },
            {
                "date": "2023-12-01 15:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36500,
                "high_price": 37500,
                "low_price": 35500,
                "close_price": 37000,
                "volume": 1500,
            },
            {
                "date": "2023-12-01 16:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37000,
                "high_price": 38000,
                "low_price": 36000,
                "close_price": 37500,
                "volume": 1700,
            },
            {
                "date": "2023-12-01 17:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37500,
                "high_price": 38500,
                "low_price": 36500,
                "close_price": 38000,
                "volume": 1900,
            },
            {
                "date": "2023-12-01 18:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38000,
                "high_price": 39000,
                "low_price": 37000,
                "close_price": 38500,
                "volume": 2100,
            },
            {
                "date": "2023-12-01 19:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38500,
                "high_price": 39500,
                "low_price": 37500,
                "close_price": 39000,
                "volume": 2300,
            },
            {
                "date": "2023-12-01 20:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39000,
                "high_price": 40000,
                "low_price": 38000,
                "close_price": 39500,
                "volume": 2500,
            },
            {
                "date": "2023-12-01 21:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39500,
                "high_price": 40500,
                "low_price": 38500,
                "close_price": 40000,
                "volume": 2700,
            },
            {
                "date": "2023-12-01 22:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40000,
                "high_price": 41000,
                "low_price": 39000,
                "close_price": 40500,
                "volume": 2900,
            },
            {
                "date": "2023-12-01 23:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40500,
                "high_price": 41500,
                "low_price": 39500,
                "close_price": 41000,
                "volume": 3100,
            },
            {
                "date": "2023-12-02 00:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41000,
                "high_price": 42000,
                "low_price": 40000,
                "close_price": 41500,
                "volume": 3300,
            },
            {
                "date": "2023-12-02 01:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41500,
                "high_price": 42500,
                "low_price": 40500,
                "close_price": 42000,
                "volume": 3500,
            },
        ]

    def test_compute_average_data_success(self):
        result = compute_average_data(self.test_data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 14)
        self.assertIn("sma_14", result.columns)
        self.assertIn("rsi_14", result.columns)
        self.assertFalse(result[["sma_14", "rsi_14"]].isnull().values.any())
        self.assertNotEqual(result["sma_14"].unique(), [39341.54068])
        self.assertNotEqual(result["rsi_14"].unique(), [100.0])

    def test_compute_average_data_insufficient_data(self):
        insufficient_data = self.test_data[:13]
        result = compute_average_data(insufficient_data)
        self.assertIsNone(result)

    def test_compute_average_data_missing_date_column(self):
        missing_date_data = [item.copy() for item in self.test_data]
        for item in missing_date_data:
            del item["date"]
        result = compute_average_data(missing_date_data)
        self.assertIsNone(result)

    def test_compute_average_data_invalid_date_format(self):
        invalid_date_data = [item.copy() for item in self.test_data]
        invalid_date_data[0]["date"] = "invalid-date"
        result = compute_average_data(invalid_date_data)
        self.assertIsNone(result)

    def test_compute_average_data_with_duplicates(self):
        duplicate_data = self.test_data + self.test_data[:5]
        result = compute_average_data(duplicate_data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 14)
        self.assertIn("sma_14", result.columns)
        self.assertIn("rsi_14", result.columns)
        self.assertFalse(result[["sma_14", "rsi_14"]].isnull().values.any())
        self.assertNotEqual(result["sma_14"].unique(), [39341.54068])
        self.assertNotEqual(result["rsi_14"].unique(), [100.0])

    def test_compute_average_data_with_nan_values(self):
        nan_data = [item.copy() for item in self.test_data]
        nan_data[0]["close_price"] = None
        result = compute_average_data(nan_data)
        self.assertIsNone(result)

    def test_compute_average_data_with_insufficient_data_after_nan_removal(self):
        nan_data = [item.copy() for item in self.test_data]
        nan_data[0]["close_price"] = None
        nan_data[1]["close_price"] = None
        nan_data[2]["close_price"] = None
        result = compute_average_data(nan_data)
        self.assertIsNone(result)

    def test_compute_average_data_wrong_structure(self):
        wrong_structure_data = [
            {"id": 1, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 12:00:00", "open_price": 35000, "high_price": 36000, "low_price": 34000, "close_price": 35500, "volume": 1000},
            {"id": 2, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 13:00:00", "open_price": 35500, "high_price": 36500, "low_price": 34500, "close_price": 36000, "volume": 1200},
            {"id": 3, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 14:00:00", "open_price": 36000, "high_price": 37000, "low_price": 35000, "close_price": 36500, "volume": 1300},
            {"id": 4, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 15:00:00", "open_price": 36500, "high_price": 37500, "low_price": 35500, "close_price": 37000, "volume": 1500},
            {"id": 5, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 16:00:00", "open_price": 37000, "high_price": 38000, "low_price": 36000, "close_price": 37500, "volume": 1700},
            {"id": 6, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 17:00:00", "open_price": 37500, "high_price": 38500, "low_price": 36500, "close_price": 38000, "volume": 1900},
            {"id": 7, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 18:00:00", "open_price": 38000, "high_price": 39000, "low_price": 37000, "close_price": 38500, "volume": 2100},
            {"id": 8, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 19:00:00", "open_price": 38500, "high_price": 39500, "low_price": 37500, "close_price": 39000, "volume": 2300},
            {"id": 9, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 20:00:00", "open_price": 39000, "high_price": 40000, "low_price": 38000, "close_price": 39500, "volume": 2500},
            {"id": 10, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 21:00:00", "open_price": 39500, "high_price": 40500, "low_price": 38500, "close_price": 40000, "volume": 2700},
            {"id": 11, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 22:00:00", "open_price": 40000, "high_price": 41000, "low_price": 39000, "close_price": 40500, "volume": 2900},
            {"id": 12, "cryptocurrency": "BTC/USDT", "date": "2023-12-01 23:00:00", "open_price": 40500, "high_price": 41500, "low_price": 39500, "close_price": 41000, "volume": 3100},
            {"id": 13, "cryptocurrency": "BTC/USDT", "date": "2023-12-02 00:00:00", "open_price": 41000, "high_price": 42000, "low_price": 40000, "close_price": 41500, "volume": 3300},
            {"id": 14, "cryptocurrency": "BTC/USDT", "date": "2023-12-02 01:00:00", "open_price": 41500, "high_price": 42500, "low_price": 40500, "close_price": 42000, "volume": 3500},
        ]
        result = compute_average_data(wrong_structure_data)
        self.assertIsNone(result)

    def test_compute_average_data_correct_sma_rsi_values(self):
        test_data = [
            {
                "date": "2023-12-01 12:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35000,
                "high_price": 36000,
                "low_price": 34000,
                "close_price": 35500,
                "volume": 1000,
            },
            {
                "date": "2023-12-01 13:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35500,
                "high_price": 36500,
                "low_price": 34500,
                "close_price": 36000,
                "volume": 1200,
            },
            {
                "date": "2023-12-01 14:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36000,
                "high_price": 37000,
                "low_price": 35000,
                "close_price": 36500,
                "volume": 1300,
            },
            {
                "date": "2023-12-01 15:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36500,
                "high_price": 37500,
                "low_price": 35500,
                "close_price": 37000,
                "volume": 1500,
            },
            {
                "date": "2023-12-01 16:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37000,
                "high_price": 38000,
                "low_price": 36000,
                "close_price": 37500,
                "volume": 1700,
            },
            {
                "date": "2023-12-01 17:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37500,
                "high_price": 38500,
                "low_price": 36500,
                "close_price": 38000,
                "volume": 1900,
            },
            {
                "date": "2023-12-01 18:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38000,
                "high_price": 39000,
                "low_price": 37000,
                "close_price": 38500,
                "volume": 2100,
            },
            {
                "date": "2023-12-01 19:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38500,
                "high_price": 39500,
                "low_price": 37500,
                "close_price": 39000,
                "volume": 2300,
            },
            {
                "date": "2023-12-01 20:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39000,
                "high_price": 40000,
                "low_price": 38000,
                "close_price": 39500,
                "volume": 2500,
            },
            {
                "date": "2023-12-01 21:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39500,
                "high_price": 40500,
                "low_price": 38500,
                "close_price": 40000,
                "volume": 2700,
            },
            {
                "date": "2023-12-01 22:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40000,
                "high_price": 41000,
                "low_price": 39000,
                "close_price": 40500,
                "volume": 2900,
            },
            {
                "date": "2023-12-01 23:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40500,
                "high_price": 41500,
                "low_price": 39500,
                "close_price": 41000,
                "volume": 3100,
            },
            {
                "date": "2023-12-02 00:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41000,
                "high_price": 42000,
                "low_price": 40000,
                "close_price": 41500,
                "volume": 3300,
            },
            {
                "date": "2023-12-02 01:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41500,
                "high_price": 42500,
                "low_price": 40500,
                "close_price": 42000,
                "volume": 3500,
            },
        ]
        result = compute_average_data(test_data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 14)
        self.assertIn("sma_14", result.columns)
        self.assertIn("rsi_14", result.columns)
        self.assertFalse(result[["sma_14", "rsi_14"]].isnull().values.any())
        self.assertNotEqual(result["sma_14"].unique(), [39341.54068])
        self.assertNotEqual(result["rsi_14"].unique(), [100.0])

    def test_compute_average_data_with_insufficient_unique_dates(self):
        insufficient_unique_dates_data = [
            {
                "date": "2023-12-01 12:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35000,
                "high_price": 36000,
                "low_price": 34000,
                "close_price": 35500,
                "volume": 1000,
            },
            {
                "date": "2023-12-01 12:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 35500,
                "high_price": 36500,
                "low_price": 34500,
                "close_price": 36000,
                "volume": 1200,
            },
            {
                "date": "2023-12-01 13:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36000,
                "high_price": 37000,
                "low_price": 35000,
                "close_price": 36500,
                "volume": 1300,
            },
            {
                "date": "2023-12-01 13:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 36500,
                "high_price": 37500,
                "low_price": 35500,
                "close_price": 37000,
                "volume": 1500,
            },
            {
                "date": "2023-12-01 14:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37000,
                "high_price": 38000,
                "low_price": 36000,
                "close_price": 37500,
                "volume": 1700,
            },
            {
                "date": "2023-12-01 14:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 37500,
                "high_price": 38500,
                "low_price": 36500,
                "close_price": 38000,
                "volume": 1900,
            },
            {
                "date": "2023-12-01 15:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38000,
                "high_price": 39000,
                "low_price": 37000,
                "close_price": 38500,
                "volume": 2100,
            },
            {
                "date": "2023-12-01 15:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 38500,
                "high_price": 39500,
                "low_price": 37500,
                "close_price": 39000,
                "volume": 2300,
            },
            {
                "date": "2023-12-01 16:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39000,
                "high_price": 40000,
                "low_price": 38000,
                "close_price": 39500,
                "volume": 2500,
            },
            {
                "date": "2023-12-01 16:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 39500,
                "high_price": 40500,
                "low_price": 38500,
                "close_price": 40000,
                "volume": 2700,
            },
            {
                "date": "2023-12-01 17:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40000,
                "high_price": 41000,
                "low_price": 39000,
                "close_price": 40500,
                "volume": 2900,
            },
            {
                "date": "2023-12-01 17:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 40500,
                "high_price": 41500,
                "low_price": 39500,
                "close_price": 41000,
                "volume": 3100,
            },
            {
                "date": "2023-12-01 18:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41000,
                "high_price": 42000,
                "low_price": 40000,
                "close_price": 41500,
                "volume": 3300,
            },
            {
                "date": "2023-12-01 18:00:00",
                "cryptocurrency": "BTC/USDT",
                "open_price": 41500,
                "high_price": 42500,
                "low_price": 40500,
                "close_price": 42000,
                "volume": 3500,
            },
        ]
        result = compute_average_data(insufficient_unique_dates_data)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
