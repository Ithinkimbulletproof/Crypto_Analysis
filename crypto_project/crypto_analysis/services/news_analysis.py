import os
import requests
from transformers import pipeline, BertForTokenClassification, BertTokenizer
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import logging
from crypto_analysis.models import NewsArticle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vader_analyzer = SentimentIntensityAnalyzer()

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

sentiment_models = [
    pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    ),
    pipeline("sentiment-analysis", model="bert-base-uncased"),
    pipeline("sentiment-analysis", model="roberta-base"),
]

tokenizer = BertTokenizer.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
model = BertForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def fetch_news(query, language="en", page_size=10):
    all_articles = []
    for page in range(1, 6):
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": NEWS_API_KEY,
                "language": language,
                "pageSize": page_size,
                "page": page,
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            news_data = response.json()
            if news_data["status"] == "ok":
                all_articles.extend(news_data["articles"])
            else:
                logger.error(
                    "Ошибка при получении новостей: %s", news_data.get("message")
                )
                break
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса: {e}")
            break
    return all_articles


def analyze_sentiment(text):
    # Получаем результаты от моделей BERT
    results = [model(text) for model in sentiment_models]
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for result in results:
        sentiment = result[0]["label"]
        sentiment_counts[sentiment] += 1

    # Определяем основной сентимент
    sentiment = max(sentiment_counts, key=sentiment_counts.get)

    # Анализируем полярность с помощью VADER
    polarity_score = vader_analyzer.polarity_scores(text)[
        "compound"
    ]  # Это числовое значение от -1 до 1

    return (
        sentiment,
        polarity_score,
    )  # Теперь функция возвращает и сентимент, и полярность


def extract_key_events(news):
    key_events = []
    for article in news:
        title = article["title"]
        description = article["description"]
        combined_text = title + " " + description
        doc = nlp(combined_text)

        key_event = {}
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "PRODUCT", "MONEY"]:
                key_event[ent.label_] = ent.text

        if key_event:
            key_events.append(
                {
                    "title": title,
                    "description": description,
                    "url": article["url"],
                    "key_event": key_event,
                }
            )

    return key_events


def gather_and_analyze_news(query="cryptocurrency"):
    news = fetch_news(query)

    if not news:
        logger.warning("Нет новостей для анализа.")
        return

    for article in news:
        title = article["title"]
        description = article["description"]
        sentiment, polarity = analyze_sentiment(title + " " + description)

        news_article = NewsArticle.objects.create(
            title=title,
            description=description,
            url=article["url"],
            sentiment=sentiment,
            polarity=polarity,
            published_at=article.get("publishedAt"),
            source=article.get("source", ""),
            language=article.get("language", "en"),
        )

        logger.info(f"Новость: {title}, Сентимент: {sentiment}, Полярность: {polarity}")

    key_events = extract_key_events(news)
    if key_events:
        for event in key_events:
            logger.info(f"Ключевое событие: {event['title']}, {event['key_event']}")
    else:
        logger.info("Нет ключевых событий.")

    return news, key_events


if __name__ == "__main__":
    query = "cryptocurrency"
    news, key_events = gather_and_analyze_news(query)
