import pandas as pd
from crypto_analysis.models import IndicatorData
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_indicator_data(selected_indicators=None):
    query = IndicatorData.objects.all()
    if selected_indicators:
        query = query.filter(indicator_name__in=selected_indicators)

    indicators = query.values("cryptocurrency", "date", "indicator_name", "value")
    df = pd.DataFrame(indicators)

    try:
        df["date"] = pd.to_datetime(df["date"], errors='raise')
    except Exception as e:
        logger.error(f"Ошибка преобразования даты: {e}")
        return pd.DataFrame()

    df_wide = df.pivot(
        index=["cryptocurrency", "date"], columns="indicator_name", values="value"
    ).reset_index()

    df_wide_numerics = df_wide.select_dtypes(include=['number'])

    df_wide_numerics.fillna(df_wide_numerics.mean(), inplace=True)

    df_wide[df_wide_numerics.columns] = df_wide_numerics

    df_wide["original_value"] = (
        df.groupby(["cryptocurrency", "date"])["value"].mean().reset_index(drop=True)
    )

    return df_wide


def short_term_indicators():
    return [
        "price_change_1d",
        "price_change_7d",
        "price_change_14d",
        "SMA_7",
        "SMA_14",
        "SMA_30",
        "volatility_7d",
        "volatility_14d",
        "volatility_30d",
        "RSI_7d",
        "RSI_14d",
        "RSI_30d",
        "CCI_7d",
        "CCI_14d",
        "ATR_14",
        "ATR_30",
        "Stochastic_7",
        "Stochastic_14",
        "MACD_12_26",
        "MACD_signal_9",
        "original_value",
    ]


def simple_prediction(crypto_data):
    indicators = short_term_indicators()
    positive_indicators = 0
    negative_indicators = 0

    for indicator in indicators:
        value = crypto_data[indicator]

        if pd.notnull(value):
            if value > 0:
                positive_indicators += 1
            else:
                negative_indicators += 1

    total_indicators = positive_indicators + negative_indicators
    if total_indicators == 0:
        return "No valid data available"

    probability_up = (positive_indicators / total_indicators) * 100
    probability_down = (negative_indicators / total_indicators) * 100

    if positive_indicators > negative_indicators:
        prediction = "Up"
    else:
        prediction = "Down"

    return prediction, probability_up, probability_down


def predict_for_all_cryptos():
    data = load_indicator_data(selected_indicators=short_term_indicators())

    cryptocurrencies = data["cryptocurrency"].unique()

    for crypto in cryptocurrencies:
        print(f"Предсказание для {crypto}")

        crypto_data = data[data["cryptocurrency"] == crypto]

        latest_data = crypto_data.iloc[-1]

        prediction, probability_up, probability_down = simple_prediction(latest_data)
        print(f"Прогноз на следующий день для {crypto}: {prediction}")
        print(f"Вероятность роста: {probability_up:.2f}%")
        print(f"Вероятность падения: {probability_down:.2f}%")
        print(f"____________________________________________")


if __name__ == "__main__":
    predict_for_all_cryptos()
