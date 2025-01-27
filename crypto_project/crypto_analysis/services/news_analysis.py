import os
import spacy
import logging
import requests
from dateutil import parser
from dotenv import load_dotenv
from django.utils import timezone
from datetime import datetime, timedelta
from crypto_analysis.models import NewsArticle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertForTokenClassification, BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vader_analyzer = SentimentIntensityAnalyzer()

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    raise ValueError("API-ключ для News API не найден. Проверьте переменные окружения.")
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)
nlp = spacy.load("en_core_web_sm")

sentiment_models = [
    pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    ),
]

tokenizer = BertTokenizer.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
model = BertForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def fetch_news(query, language="en", page_size=100, days=3):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    all_articles = []
    page = 1
    while True:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": NEWS_API_KEY,
                "language": language,
                "pageSize": page_size,
                "page": page,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            news_data = response.json()
            if news_data["status"] == "ok":
                all_articles.extend(news_data["articles"])

                if len(news_data["articles"]) < page_size:
                    break
                page += 1
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
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    result = sentiment_analyzer(text)[0]
    sentiment = result["label"]
    sentiment_counts[sentiment] += 1

    polarity_score = vader_analyzer.polarity_scores(text)["compound"]

    return sentiment, polarity_score


def extract_key_events(news):
    key_events = []
    for article in news:
        title = article["title"]
        description = article["description"] or ""
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


def gather_and_analyze_news(query="cryptocurrency OR BTC OR ETH OR crypto market"):
    news = fetch_news(query)

    if not news:
        logger.warning("Нет новостей для анализа.")
        return

    for article in news:
        title = article["title"]
        description = article["description"] or ""
        url = article["url"]
        published_at = article.get("publishedAt")
        source = article.get("source", "")
        language = article.get("language", "en")

        if published_at:
            published_at = parser.parse(published_at)
            published_at = timezone.localtime(published_at)

        if NewsArticle.objects.filter(url=url).exists():
            logger.info(f"Новость уже существует: {title}")
            continue

        sentiment, polarity = analyze_sentiment(title + " " + description)

        if sentiment == "NEGATIVE" and polarity < -0.5:
            logger.info(f"Отфильтрована негативная новость: {title}")
            continue

        news_article = NewsArticle.objects.create(
            title=title,
            description=description,
            url=url,
            sentiment=sentiment,
            polarity=polarity,
            published_at=published_at,
            source=source,
            language=language,
        )

        logger.info(f"Новость: {title}, Сентимент: {sentiment}, Полярность: {polarity}")

    key_events = extract_key_events(news)
    if key_events:
        for event in key_events:
            logger.info(f"Ключевое событие: {event['title']}, {event['key_event']}")
    else:
        logger.info("Нет ключевых событий.")

    return news, key_events
