import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from crypto_analysis.ml_models.stacking import predict_and_save


def run_predictions():
    load_dotenv()
    cryptopairs = os.getenv("CRYPTOPAIRS")
    if not cryptopairs:
        raise ValueError(
            "Переменная окружения CRYPTOPAIRS не установлена. Укажите её в файле .env"
        )
    currency = cryptopairs.strip()

    with open("features_list.json", "r") as f:
        features_list = json.load(f)

    stacking_model = joblib.load(
        f"models/stacking_unified_{currency.replace('/', '_')}.pkl"
    )
    gru_model_1h = joblib.load(f"models/gru_1h_{currency.replace('/', '_')}.pkl")
    gru_model_24h = joblib.load(f"models/gru_24h_{currency.replace('/', '_')}.pkl")
    transformer_model_1h = joblib.load(
        f"models/transformer_1h_{currency.replace('/', '_')}.pkl"
    )
    transformer_model_24h = joblib.load(
        f"models/transformer_24h_{currency.replace('/', '_')}.pkl"
    )
    xgboost_model_1h = joblib.load(
        f"models/xgboost_1h_{currency.replace('/', '_')}.pkl"
    )
    xgboost_model_24h = joblib.load(
        f"models/xgboost_24h_{currency.replace('/', '_')}.pkl"
    )
    lightgbm_model_1h = joblib.load(
        f"models/lightgbm_1h_{currency.replace('/', '_')}.pkl"
    )
    lightgbm_model_24h = joblib.load(
        f"models/lightgbm_24h_{currency.replace('/', '_')}.pkl"
    )
    prophet_model = joblib.load(
        f"models/prophet_{currency.replace('/', '_')}_{currency.replace('/', '_')}.pkl"
    )
    arima_model = joblib.load(
        f"models/arima_{currency.replace('/', '_')}_{currency.replace('/', '_')}.pkl"
    )

    models = {
        "gru_1h": gru_model_1h,
        "gru_24h": gru_model_24h,
        "transformer_1h": transformer_model_1h,
        "transformer_24h": transformer_model_24h,
        "xgboost_1h": xgboost_model_1h,
        "xgboost_24h": xgboost_model_24h,
        "lightgbm_1h": lightgbm_model_1h,
        "lightgbm_24h": lightgbm_model_24h,
        "prophet": prophet_model,
        "arima": arima_model,
    }

    unified_data_file = f"data_exports/unified_data_{currency.replace('/', '_')}.csv"
    if not os.path.exists(unified_data_file):
        raise FileNotFoundError(f"Файл {unified_data_file} не найден.")

    X_actual = pd.read_csv(unified_data_file, low_memory=False)
    print(
        f"X_actual загружен: {X_actual.shape}, диапазон дат: {X_actual['date'].min()} - {X_actual['date'].max()}"
    )

    X_actual["date"] = pd.to_datetime(X_actual["date"], errors="coerce")
    if X_actual["date"].isna().all():
        raise ValueError(
            "Не удалось преобразовать столбец 'date' в datetime. Проверьте формат данных."
        )

    X_actual.columns = X_actual.columns.str.replace(" ", "_")

    if "cryptocurrency" in X_actual.columns:
        print(
            f"Уникальные значения cryptocurrency: {X_actual['cryptocurrency'].unique()}"
        )
        print(f"Фильтрация по cryptocurrency == '{currency}'")
        X_actual = X_actual[X_actual["cryptocurrency"] == currency]
        print(f"После фильтрации по cryptocurrency: {X_actual.shape}")
        if X_actual.empty:
            raise ValueError(
                f"Нет данных для cryptocurrency == '{currency}'. Проверьте файл или значение CRYPTOPAIRS."
            )
    else:
        print("Столбец 'cryptocurrency' отсутствует. Пропускаем фильтрацию.")

    X_actual = X_actual.dropna(subset=["close_price"])
    print(f"После удаления NaN в close_price: {X_actual.shape}")

    if X_actual.empty:
        raise ValueError(
            "X_actual пуст после начальной обработки. Проверьте данные в файле."
        )

    if "close_price_24h" not in X_actual.columns:
        X_actual["close_price_1h"] = X_actual["close_price"].shift(-4)
        X_actual["close_price_24h"] = X_actual["close_price"].shift(-96)
    if "volume.1" not in X_actual.columns:
        X_actual["volume.1"] = X_actual["volume"]
    if "BBANDS_Middle.1" not in X_actual.columns:
        X_actual["BBANDS_Middle.1"] = X_actual["BBANDS_Middle"]
    if "VWAP.1" not in X_actual.columns:
        X_actual["VWAP.1"] = X_actual["VWAP"]
    if "hour" not in X_actual.columns:
        X_actual["hour"] = X_actual["date"].dt.hour
    if "day_of_week" not in X_actual.columns:
        X_actual["day_of_week"] = X_actual["date"].dt.dayofweek

    cutoff_7d = X_actual["date"].max() - pd.Timedelta(days=7)
    print(f"Cutoff date: {cutoff_7d}")
    X_daily = X_actual[X_actual["date"] >= cutoff_7d]
    print(f"X_daily после фильтрации за 7 дней: {X_daily.shape}")

    if X_daily.empty:
        raise ValueError(
            f"Нет данных за последние 7 дней (с {cutoff_7d} до {X_actual['date'].max()}). Обновите unified_data_{currency}.csv."
        )

    X_inference = X_daily.copy()

    temp = X_inference[X_inference["date"].dt.minute == 0]
    if not temp.empty:
        X_inference = temp
        print(f"X_inference после фильтрации по minute == 0: {X_inference.shape}")
    else:
        print("Warning: Нет строк с minute == 0, используем всю выборку за 7 дней")
        print(f"X_inference без фильтрации по минутам: {X_inference.shape}")

    if X_inference.empty:
        raise ValueError("X_inference пуст после всех фильтраций.")

    X_inference_numeric = X_inference.reindex(columns=features_list, fill_value=0)
    print(f"X_inference_numeric после reindex: {X_inference_numeric.shape}")

    current_date = datetime.now()
    symbols = [currency]
    final_pred, prediction = predict_and_save(
        models,
        stacking_model,
        X_inference_numeric,
        current_date,
        symbols,
        raw_data=X_inference,
    )

    price_1h = final_pred[-1, 0]
    price_24h = final_pred[-1, 1]
    print(f"✅ Для {symbols} предсказание: на 1h = {price_1h}, на 24h = {price_24h}")
