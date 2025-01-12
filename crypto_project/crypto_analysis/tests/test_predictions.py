import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from crypto_analysis.services.predictions import analyze_and_update


class TestAnalyzeAndUpdate(unittest.TestCase):

    @patch(
        "crypto_analysis.models.MarketData.objects.filter"
    )  # Патчим запрос к базе данных
    @patch(
        "crypto_analysis.services.data_processor.prepare_data_for_analysis"
    )  # Патчим функцию подготовки данных
    @patch(
        "crypto_analysis.services.data_processor.get_features_and_labels"
    )  # Патчим функцию извлечения признаков
    @patch(
        "crypto_analysis.services.data_processor.scale_and_resample_data"
    )  # Патчим ресэмплинг
    @patch(
        "crypto_analysis.services.machine_learning.train_and_evaluate_models"
    )  # Патчим функцию обучения
    @patch(
        "crypto_analysis.services.utils.save_predictions"
    )  # Патчим сохранение предсказаний
    def test_analyze_and_update(
        self,
        mock_save_predictions,
        mock_train_and_evaluate_models,
        mock_scale_and_resample_data,
        mock_get_features_and_labels,
        mock_prepare_data_for_analysis,
        mock_filter,
    ):
        # Заглушки для всех внешних зависимостей

        # Патчим запрос к базе данных
        mock_filter.return_value.all.return_value = [
            {
                "date": datetime.now(timezone.utc),
                "open": 50000,
                "close": 50500,
                "high": 51000,
                "low": 49500,
                "volume": 1000,
            }
        ]  # Подставляем фиктивные данные, имитирующие запрос к базе

        # Патчим подготовку данных для анализа
        mock_prepare_data_for_analysis.return_value = [
            {
                "date": datetime.now(timezone.utc),
                "open": 50000,
                "close": 50500,
                "high": 51000,
                "low": 49500,
                "volume": 1000,
            }
        ]  # Подставляем подготовленные данные

        # Патчим извлечение признаков и меток
        mock_get_features_and_labels.return_value = ([1, 2, 3], [0])  # Признаки и метки

        # Патчим масштабирование и ресэмплинг
        mock_scale_and_resample_data.return_value = (
            [1, 2, 3],
            [0],
        )  # Масштабированные данные и ресэмплированные метки

        # Патчим обучение моделей
        mock_train_and_evaluate_models.return_value = (
            1,
            1,
            [1],
            [0.9],
        )  # Предсказания и вероятности

        # Патчим сохранение предсказаний
        mock_save_predictions.return_value = (
            None  # Просто симуляция вызова без результатов
        )

        # Вызываем тестируемую функцию
        analyze_and_update()

        # Проверяем, что все функции были вызваны один раз
        mock_filter.assert_called_once()  # Проверяем, что запрос к базе был сделан один раз
        mock_prepare_data_for_analysis.assert_called_once()  # Проверяем, что подготовка данных была вызвана
        mock_get_features_and_labels.assert_called_once()  # Проверяем, что признаки и метки были извлечены
        mock_scale_and_resample_data.assert_called_once()  # Проверяем, что данные были масштабированы и ресэмплированы
        mock_train_and_evaluate_models.assert_called_once()  # Проверяем, что модели были обучены
        mock_save_predictions.assert_called_once()  # Проверяем, что предсказания были сохранены


if __name__ == "__main__":
    unittest.main()
