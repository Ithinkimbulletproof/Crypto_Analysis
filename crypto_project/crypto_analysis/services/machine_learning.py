import time
import logging
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_models():
    logger.info("Инициализация моделей для обучения.")
    models = {
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=42
        ),
        "Random Forest": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=10000, solver="lbfgs", random_state=42
        ),
    }
    logger.info("Модели успешно инициализированы.")
    return models


def generate_param_grid(model_name):
    grids = {
        "Gradient Boosting": {
            "max_iter": [100, 200],
            "learning_rate": [0.01, 0.05],
            "max_depth": [3],
        },
        "Random Forest": {
            "n_estimators": [100, 150],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
        },
        "Logistic Regression": {"C": [0.01, 0.1]},
    }
    return grids.get(model_name, {})


def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    logger.info("Начало обучения и оценки моделей")
    tscv = TimeSeriesSplit(n_splits=3)
    all_predictions = []
    all_probabilities = []
    total_predictions = 0
    correct_predictions = 0
    for model_name, model in models.items():
        logger.info(f"Обработка модели: {model_name}")
        param_grid = generate_param_grid(model_name)
        try:
            if param_grid:
                logger.info(f"Параметры для модели {model_name}: {param_grid}")
                model = GridSearchCV(
                    model, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=1
                )
            logger.info(f"Обучение модели {model_name}")
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            logger.info(f"Время обучения модели: {end_time - start_time:.2f} секунд")
            predictions = model.predict(X_test)
            probabilities = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )
            all_predictions.extend(predictions)
            all_probabilities.extend(
                probabilities
                if probabilities is not None
                else [None] * len(predictions)
            )
            logger.info(
                f"Получено {len(predictions)} предсказаний для модели {model_name}"
            )
            if probabilities is not None:
                logger.info(
                    f"Получено {len(probabilities)} вероятностей для модели {model_name}"
                )
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            logger.info(
                f"Точность модели {model_name}: {accuracy:.2%}, F1-score: {f1:.2%}"
            )
            total_predictions += len(predictions)
            correct_predictions += sum(predictions == y_test)
        except Exception as e:
            logger.error(f"Ошибка при обработке модели {model_name}: {e}")
            logger.error(traceback.format_exc())
    logger.info("Завершено обучение и оценка моделей")
    return total_predictions, correct_predictions, all_predictions, all_probabilities
