import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        df = (
            df.groupby("news_hour")[
                ["vader_compound", "bert_positive", "combined_score"]
            ]
            .mean()
            .reset_index()
        )
    return df


def get_key_entities_df():
    qs = KeyEntity.objects.all().values("article__published_at", "entity_type", "text")
    df = pd.DataFrame.from_records(qs)
    if not df.empty:
        df.rename(columns={"article__published_at": "date"}, inplace=True)
        df["news_hour"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("h")
        df_entities = (
            df.groupby("news_hour")
            .agg({"entity_type": lambda x: list(x), "text": lambda x: list(x)})
            .reset_index()
        )
        df_entities["entities"] = df_entities.apply(
            lambda row: json.dumps(
                {
                    row["entity_type"][i]: row["text"][i]
                    for i in range(len(row["entity_type"]))
                },
                ensure_ascii=False,
            ),
            axis=1,
        )
        df_entities = df_entities.drop(columns=["entity_type", "text"])
        return df_entities
    return pd.DataFrame()


def build_unified_dataframe():
    df_market = get_market_data_df()
    df_indicators = get_indicators_df()
    df_news = get_news_sentiment_df()
    df_entities = get_key_entities_df()

    if df_market.empty:
        return pd.DataFrame()

    df_merged = pd.merge(
        df_market,
        df_indicators,
        on=["cryptocurrency", "date", "news_hour"],
        how="left",
        suffixes=("", "_ind"),
    )

    if not df_news.empty:
        df_merged = pd.merge(
            df_merged, df_news, on="news_hour", how="left", suffixes=("", "_news")
        )

    if not df_entities.empty:
        df_unified = pd.merge(df_merged, df_entities, on="news_hour", how="left")
    else:
        df_unified = df_merged.copy()

    return df_unified


def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


def preprocessing_data(df):
    if "date" not in df.columns:
        if "date_x" in df.columns:
            df = df.rename(columns={"date_x": "date"})
        elif "date_y" in df.columns:
            df = df.rename(columns={"date_y": "date"})
        else:
            raise KeyError("Не найдена колонка 'date' в DataFrame.")

    df = df.sort_values("date").reset_index(drop=True)
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek

    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    df = remove_outliers_iqr(df, price_cols)

    volume_cols = ["volume"]
    sentiment_cols = ["vader_compound", "bert_positive", "combined_score"]
    numeric_cols = price_cols + volume_cols + sentiment_cols

    additional_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in additional_numeric:
        if col not in numeric_cols and col not in ["close_price_1h", "close_price_24h"]:
            numeric_cols.append(col)

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df["close_price_1h"] = df["close_price"].shift(-4)
    df["close_price_24h"] = df["close_price"].shift(-96)
    df = df.dropna(subset=["close_price_1h", "close_price_24h"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    features_to_scale = numeric_cols.copy()
    if "close_price_1h" in features_to_scale:
        features_to_scale.remove("close_price_1h")
    if "close_price_24h" in features_to_scale:
        features_to_scale.remove("close_price_24h")

    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(df_train[features_to_scale])
    X_test_minmax = minmax_scaler.transform(df_test[features_to_scale])

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(df_train[features_to_scale])
    X_test_std = std_scaler.transform(df_test[features_to_scale])

    y_train = df_train[["close_price_1h", "close_price_24h"]]
    y_test = df_test[["close_price_1h", "close_price_24h"]]

    target_scaler_minmax = MinMaxScaler()
    y_train_minmax_scaled = target_scaler_minmax.fit_transform(y_train)
    y_test_minmax_scaled = target_scaler_minmax.transform(y_test)

    target_scaler_std = StandardScaler()
    y_train_std_scaled = target_scaler_std.fit_transform(y_train)
    y_test_std_scaled = target_scaler_std.transform(y_test)

    df_minmax_full = df.copy()
    df_minmax_full[features_to_scale] = minmax_scaler.transform(
        df_minmax_full[features_to_scale]
    )
    df_std_full = df.copy()
    df_std_full[features_to_scale] = std_scaler.transform(
        df_std_full[features_to_scale]
    )

    processed_data = {
        "df_original": df,
        "df_train_minmax": pd.concat(
            [
                df_train.reset_index(drop=True),
                pd.DataFrame(X_train_minmax, columns=features_to_scale),
            ],
            axis=1,
        ),
        "df_test_minmax": pd.concat(
            [
                df_test.reset_index(drop=True),
                pd.DataFrame(X_test_minmax, columns=features_to_scale),
            ],
            axis=1,
        ),
        "X_train_minmax": pd.DataFrame(X_train_minmax, columns=features_to_scale),
        "X_test_minmax": pd.DataFrame(X_test_minmax, columns=features_to_scale),
        "y_train_minmax": y_train,
        "y_test_minmax": y_test,
        "y_train_minmax_scaled": pd.DataFrame(
            y_train_minmax_scaled, columns=["close_price_1h", "close_price_24h"]
        ),
        "y_test_minmax_scaled": pd.DataFrame(
            y_test_minmax_scaled, columns=["close_price_1h", "close_price_24h"]
        ),
        "df_train_std": pd.concat(
            [
                df_train.reset_index(drop=True),
                pd.DataFrame(X_train_std, columns=features_to_scale),
            ],
            axis=1,
        ),
        "df_test_std": pd.concat(
            [
                df_test.reset_index(drop=True),
                pd.DataFrame(X_test_std, columns=features_to_scale),
            ],
            axis=1,
        ),
        "X_train_std": pd.DataFrame(X_train_std, columns=features_to_scale),
        "X_test_std": pd.DataFrame(X_test_std, columns=features_to_scale),
        "y_train_std": y_train,
        "y_test_std": y_test,
        "y_train_std_scaled": pd.DataFrame(
            y_train_std_scaled, columns=["close_price_1h", "close_price_24h"]
        ),
        "y_test_std_scaled": pd.DataFrame(
            y_test_std_scaled, columns=["close_price_1h", "close_price_24h"]
        ),
        "df_minmax": df_minmax_full,
        "df_std": df_std_full,
        "minmax_scaler": minmax_scaler,
        "std_scaler": std_scaler,
        "target_scaler_minmax": target_scaler_minmax,
        "target_scaler_std": target_scaler_std,
    }
    return processed_data, features_to_scale


def save_csv_files_by_currency():
    df = build_unified_dataframe()
    if df.empty:
        raise ValueError("Объединённый DataFrame пуст!")

    save_path = os.path.join(os.getcwd(), "data_exports")
    os.makedirs(save_path, exist_ok=True)

    currencies = df["cryptocurrency"].unique()

    for currency in currencies:
        safe_currency = str(currency).strip().replace("/", "_").replace("\\", "_")

        df_currency = df[df["cryptocurrency"] == currency].copy()

        processed_data, features_to_scale = preprocessing_data(df_currency)

        unified_file = os.path.join(save_path, f"unified_data_{safe_currency}.csv")
        df_currency.to_csv(unified_file, index=False)

        processed_train_minmax_file = os.path.join(
            save_path, f"processed_train_minmax_{safe_currency}.csv"
        )
        processed_data["df_train_minmax"].to_csv(
            processed_train_minmax_file, index=False
        )

        processed_test_minmax_file = os.path.join(
            save_path, f"processed_test_minmax_{safe_currency}.csv"
        )
        processed_data["df_test_minmax"].to_csv(processed_test_minmax_file, index=False)

        processed_train_std_file = os.path.join(
            save_path, f"processed_train_std_{safe_currency}.csv"
        )
        processed_data["df_train_std"].to_csv(processed_train_std_file, index=False)

        processed_test_std_file = os.path.join(
            save_path, f"processed_test_std_{safe_currency}.csv"
        )
        processed_data["df_test_std"].to_csv(processed_test_std_file, index=False)

    return save_path
