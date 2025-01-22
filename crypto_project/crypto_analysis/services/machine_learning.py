from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from django.db import transaction
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from crypto_analysis.models import IndicatorData, ShortTermCryptoPrediction, LongTermCryptoPrediction


def load_indicator_data():
    indicators = IndicatorData.objects.all().values(
        'cryptocurrency', 'date', 'indicator_name', 'value'
    )
    df = pd.DataFrame(indicators)
    df['date'] = pd.to_datetime(df['date'])
    df = df.pivot_table(index=['cryptocurrency', 'date'], columns='indicator_name', values='value').reset_index()
    return df


def short_term_forecasting(data):
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    }
    predictions = []
    for model_name, model in models.items():
        for crypto, crypto_data in data.groupby('cryptocurrency'):
            crypto_data = crypto_data.sort_values('date')
            X, y = prepare_data_for_ml(crypto_data)
            model.fit(X, y)
            y_pred = model.predict([X[-1]])[0]
            predictions.append({
                'cryptocurrency': crypto,
                'predicted_change': y_pred,
                'model_type': model_name,
                'confidence': calculate_confidence(model, X, y)
            })
    return predictions


def long_term_forecasting(data):
    models = {
        'Prophet': Prophet(),
        'ARIMA': ARIMA,
        'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    }
    predictions = []
    for model_name, model in models.items():
        for crypto, crypto_data in data.groupby('cryptocurrency'):
            crypto_data = crypto_data.sort_values('date')
            X, y = prepare_data_for_ml(crypto_data)
            if model_name == 'Prophet':
                df_prophet = prepare_data_for_prophet(crypto_data)
                model.fit(df_prophet)
                forecast = model.predict(df_prophet)
                y_pred = forecast['yhat'][-1]
            elif model_name == 'ARIMA':
                model = model(crypto_data['value'], order=(5, 1, 0))
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=1)[0]
            elif model_name == 'MLP':
                model.fit(X, y)
                y_pred = model.predict([X[-1]])[0]
            predictions.append({
                'cryptocurrency': crypto,
                'predicted_change': y_pred,
                'model_type': model_name,
                'confidence': calculate_confidence(model, X, y)
            })
    return predictions


def prepare_data_for_ml(data):
    X = data.drop(['cryptocurrency', 'date'], axis=1).values[:-1]
    y = data['value'].values[1:]
    return X, y


def prepare_data_for_prophet(data):
    df_prophet = data[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
    return df_prophet


def calculate_confidence(model, X, y):
    if hasattr(model, 'score'):
        return model.score(X, y)
    return 1.0


@transaction.atomic
def save_predictions(predictions, is_short_term=True):
    model = ShortTermCryptoPrediction if is_short_term else LongTermCryptoPrediction
    for pred in predictions:
        model.objects.update_or_create(
            cryptocurrency_pair=pred['cryptocurrency'],
            prediction_date=pd.Timestamp.now().date(),
            model_type=pred['model_type'],
            defaults={
                'predicted_price_change': pred['predicted_change'],
                'predicted_close': pred.get('predicted_close', 0),
                'confidence_level': pred['confidence']
            }
        )


def run_forecasting():
    data = load_indicator_data()
    short_predictions = short_term_forecasting(data)
    save_predictions(short_predictions, is_short_term=True)
    long_predictions = long_term_forecasting(data)
    save_predictions(long_predictions, is_short_term=False)
