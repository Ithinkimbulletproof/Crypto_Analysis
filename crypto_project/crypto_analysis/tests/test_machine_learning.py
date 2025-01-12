import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from crypto_analysis.services.machine_learning import (
    get_models,
    generate_param_grid,
    train_and_evaluate_models,
)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.models = get_models()

    def test_get_models(self):
        models = get_models()
        self.assertIn("Gradient Boosting", models)
        self.assertIn("Random Forest", models)
        self.assertIn("Logistic Regression", models)
        self.assertIsInstance(
            models["Gradient Boosting"], HistGradientBoostingClassifier
        )
        self.assertIsInstance(models["Random Forest"], ExtraTreesClassifier)
        self.assertIsInstance(models["Logistic Regression"], LogisticRegression)

    def test_generate_param_grid(self):
        gb_params = generate_param_grid("Gradient Boosting")
        rf_params = generate_param_grid("Random Forest")
        lr_params = generate_param_grid("Logistic Regression")
        self.assertEqual(gb_params["max_iter"], [100, 200])
        self.assertEqual(rf_params["n_estimators"], [100, 150])
        self.assertEqual(lr_params["C"], [0.01, 0.1])
        self.assertEqual(generate_param_grid("Unknown Model"), {})

    def test_train_and_evaluate_models(self):
        total_predictions, correct_predictions, all_predictions, all_probabilities = (
            train_and_evaluate_models(
                self.models, self.X_train, self.y_train, self.X_test, self.y_test
            )
        )

        model_predictions = {model_name: [] for model_name in self.models}
        model_probabilities = {model_name: [] for model_name in self.models}

        for i, model_name in enumerate(self.models):
            model_predictions[model_name] = all_predictions[
                i * len(self.X_test) : (i + 1) * len(self.X_test)
            ]
            model_probabilities[model_name] = all_probabilities[
                i * len(self.X_test) : (i + 1) * len(self.X_test)
            ]

            self.assertEqual(len(model_predictions[model_name]), len(self.X_test))
            self.assertEqual(len(model_probabilities[model_name]), len(self.X_test))

        self.assertGreaterEqual(total_predictions, len(self.X_test))
        self.assertGreaterEqual(correct_predictions, 0)

        accuracy = correct_predictions / total_predictions
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_empty_models(self):
        total_predictions, correct_predictions, all_predictions, all_probabilities = (
            train_and_evaluate_models(
                {}, self.X_train, self.y_train, self.X_test, self.y_test
            )
        )
        self.assertEqual(total_predictions, 0)
        self.assertEqual(correct_predictions, 0)
        self.assertEqual(len(all_predictions), 0)
        self.assertEqual(len(all_probabilities), 0)


if __name__ == "__main__":
    unittest.main()
