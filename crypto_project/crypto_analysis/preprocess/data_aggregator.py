import pandas as pd
from crypto_analysis.models import MarketData, IndicatorData, SentimentData, KeyEntity


def get_market_data_df():
    qs = MarketData.objects.all().values(
        "cryptocurrency",
        "date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
    )
    df = pd.DataFrame.from_records(qs)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["news_hour"] = df["date"].dt.floor("h")
    return df


def get_indicators_df():
    qs = IndicatorData.objects.all().values(
        "cryptocurrency", "date", "indicator_name", "value"
    )
    df = pd.DataFrame.from_records(qs)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["news_hour"] = df["date"].dt.floor("h")
        df = df.pivot_table(
            index=["cryptocurrency", "date", "news_hour"],
            columns="indicator_name",
            values="value",
        ).reset_index()
        df = df.dropna(axis=1, how="all")
    return df


def get_news_sentiment_df():
    qs = SentimentData.objects.all().values(
        "article__published_at",
        "vader_compound",
        "bert_positive",
        "combined_score",
        "article__title",
    )
    df = pd.DataFrame.from_records(qs)
    if not df.empty:
        df.rename(columns={"article__published_at": "date"}, inplace=True)
        df["news_hour"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("h")

        df_sentiment = (
            df.groupby("news_hour")[
                ["vader_compound", "bert_positive", "combined_score"]
            ]
            .mean()
            .reset_index()
        )
        df = df_sentiment
    return df


def get_key_entities_df():
    qs = KeyEntity.objects.all().values("article__published_at", "entity_type", "text")
    df = pd.DataFrame.from_records(qs)
    if not df.empty:
        df.rename(columns={"article__published_at": "date"}, inplace=True)
        df["news_hour"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("h")
        df_entities = (
            df.groupby(["date", "news_hour"])
            .agg({"entity_type": lambda x: list(x), "text": lambda x: list(x)})
            .reset_index()
        )
        df_entities["entities"] = df_entities.apply(
            lambda row: {
                row["entity_type"][i]: row["text"][i]
                for i in range(len(row["entity_type"]))
            },
            axis=1,
        )
        df_entities = df_entities.drop(columns=["entity_type", "text"])
    return df_entities


def build_unified_dataframe():
    df_market = get_market_data_df()
    df_indicators = get_indicators_df()
    df_news = get_news_sentiment_df()
    df_entities = get_key_entities_df()

    print("Размерность MarketData:", df_market.shape)
    print("Размерность IndicatorData (после pivot):", df_indicators.shape)
    print("Размерность NewsSentiment:", df_news.shape)
    print("Размерность KeyEntities:", df_entities.shape)

    print("Уникальные news_hour в MarketData:", df_market["news_hour"].unique())
    print("Уникальные news_hour в NewsSentiment:", df_news["news_hour"].unique())

    if df_market.empty:
        return pd.DataFrame()

    df_merged = pd.merge(
        df_market,
        df_indicators,
        on=["cryptocurrency", "date", "news_hour"],
        how="left",
        suffixes=("", "_ind"),
    )
    print("Размерность после объединения MarketData и IndicatorData:", df_merged.shape)

    if not df_news.empty:
        df_merged = pd.merge(
            df_merged, df_news, on="news_hour", how="left", suffixes=("", "_news")
        )

    if not df_entities.empty:
        df_unified = pd.merge(df_merged, df_entities, on="news_hour", how="left")
    else:
        df_unified = df_merged.copy()

    return df_unified


if __name__ == "__main__":
    df = build_unified_dataframe()
    print("Объединенный DataFrame:")
    print(df.head())
    print(f"Размерность итогового DataFrame: {df.shape}")
    df.to_csv("unified_data.csv", index=False)
