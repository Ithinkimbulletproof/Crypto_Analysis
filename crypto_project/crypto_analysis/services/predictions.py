import pandas as pd
from django.utils.dateparse import parse_date
from crypto_analysis.models import ShortTermCryptoPrediction, LongTermCryptoPrediction


def import_predictions_from_csv(file_path, model_type="short_term"):
    df = pd.read_csv(file_path)

    if model_type == "short_term":
        model_class = ShortTermCryptoPrediction
    elif model_type == "long_term":
        model_class = LongTermCryptoPrediction
    else:
        raise ValueError(
            "Unknown model type. Choose between 'short_term' or 'long_term'."
        )

    for _, row in df.iterrows():
        cryptocurrency_pair = row["cryptocurrency"]
        prediction_date = parse_date(row["date"])
        predicted_price_change = row["predicted_change"]
        predicted_close = row["predicted_close"]
        confidence_level = 1.0

        if model_class.objects.filter(
            cryptocurrency_pair=cryptocurrency_pair,
            prediction_date=prediction_date,
            model_type=model_type,
        ).exists():
            print(
                f"Prediction for {cryptocurrency_pair} on {prediction_date} already exists. Skipping."
            )
            continue

        prediction = model_class(
            cryptocurrency_pair=cryptocurrency_pair,
            prediction_date=prediction_date,
            predicted_price_change=predicted_price_change,
            predicted_close=predicted_close,
            model_type=model_type,
            confidence_level=confidence_level,
        )
        prediction.save()
        print(f"Prediction for {cryptocurrency_pair} on {prediction_date} saved.")


def import_short_term_predictions(file_path):
    import_predictions_from_csv(file_path, model_type="short_term")


def import_long_term_predictions(file_path):
    import_predictions_from_csv(file_path, model_type="long_term")
