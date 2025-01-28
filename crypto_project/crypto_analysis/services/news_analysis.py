import spacy
import logging
import requests
import random
from dateutil import parser
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from crypto_analysis.models import NewsArticle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vader_analyzer = SentimentIntensityAnalyzer()
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

NEWS_SOURCES = [
    {
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "parser": "cointelegraph",
        "backup_urls": [
            "https://cointelegraph.com/rss/tag/bitcoin",
            "https://cointelegraph.com/rss/tag/ethereum",
        ],
    },
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "parser": "coindesk",
    },
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
]

sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)
nlp = spacy.load("en_core_web_sm")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    before_sleep=lambda *_: logger.warning("Retrying request..."),
)
def fetch_news(days=3, max_results=50):
    articles = []

    for source in NEWS_SOURCES:
        try:
            new_articles = parse_source(
                source["url"],
                source["parser"],
                days,
                max_results // len(NEWS_SOURCES),
                headers=source.get("headers", {}),
            )
            articles.extend(new_articles)

            if not new_articles and "backup_urls" in source:
                for backup_url in source["backup_urls"]:
                    try:
                        new_articles = parse_source(
                            backup_url,
                            source["parser"],
                            days,
                            max_results // len(NEWS_SOURCES),
                            headers=source.get("headers", {}),
                        )
                        articles.extend(new_articles)
                        if new_articles:
                            break
                    except Exception as e:
                        logger.error(f"Backup source error: {str(e)}")

        except Exception as e:
            logger.error(f"Source {source['name']} error: {str(e)}")

    filtered = [a for a in articles if is_recent(a["publishedAt"], days)]
    return sorted(filtered[:max_results], key=lambda x: x["publishedAt"], reverse=True)


def parse_source(url, parser_type, days, max_per_source, headers=None):
    try:
        session = requests.Session()
        final_headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/xml, text/xml;q=0.9, */*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": url.split("/feed")[0],
            **(headers or {}),
        }

        response = session.get(
            url, headers=final_headers, timeout=15, allow_redirects=True
        )
        response.raise_for_status()

        min_date = datetime.now(timezone.utc) - timedelta(days=days)
        return parse_feed(response.content, parser_type, min_date, max_per_source)
    except Exception as e:
        raise Exception(f"Parsing error: {str(e)}")


def parse_feed(xml_content, parser_type, min_date, max_results):
    soup = BeautifulSoup(xml_content, "xml")
    articles = []

    for item in soup.find_all("item")[:max_results]:
        try:
            parser_func = globals().get(f"parse_{parser_type}")
            if parser_func:
                article = parser_func(item)
                if article and parser.parse(article["publishedAt"]) > min_date:
                    articles.append(article)
        except Exception as e:
            logger.error(f"Article parsing error: {str(e)}")

    return articles


def parse_cointelegraph(item):
    pub_date = parser.parse(item.pubDate.text)
    if pub_date.tzinfo is None:
        pub_date = pub_date.replace(tzinfo=timezone.utc)
    return {
        "title": item.title.text.strip(),
        "description": clean_html(item.description.text),
        "url": item.link.text.strip(),
        "publishedAt": pub_date.isoformat(),
        "source": "Cointelegraph",
        "language": "en",
    }


def parse_coindesk(item):
    pub_date = parser.parse(item.pubDate.text)
    if pub_date.tzinfo is None:
        pub_date = pub_date.replace(tzinfo=timezone.utc)
    return {
        "title": item.title.text.strip(),
        "description": clean_html(item.description.text),
        "url": item.link.text.strip(),
        "publishedAt": pub_date.isoformat(),
        "source": "CoinDesk",
        "language": "en",
    }


def clean_html(text):
    if text.strip().startswith(("<", "&lt;")):
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return text.strip()


def is_recent(pub_date, days):
    try:
        parsed_date = parser.parse(pub_date)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return parsed_date > cutoff
    except Exception as e:
        logger.error(f"Date parsing error: {str(e)}")
        return False


def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]
    vader_score = vader_analyzer.polarity_scores(text)["compound"]
    return result["label"], vader_score


def extract_key_events(news):
    key_events = []
    for article in news:
        doc = nlp(article["title"] + " " + article["description"])
        entities = {
            ent.label_: ent.text
            for ent in doc.ents
            if ent.label_ in ["ORG", "GPE", "MONEY", "PRODUCT"]
        }
        if entities:
            key_events.append(
                {
                    "title": article["title"],
                    "url": article["url"],
                    "entities": entities,
                    "published_at": article["publishedAt"],
                }
            )
    return key_events


def gather_and_analyze_news():
    news = fetch_news(days=3, max_results=100)

    if not news:
        logger.warning("No news to analyze")
        return None, None

    for article in news:
        try:
            if NewsArticle.objects.filter(url=article["url"]).exists():
                continue

            sentiment, polarity = analyze_sentiment(
                f"{article['title']} {article['description']}"
            )

            if sentiment == "NEGATIVE" and polarity < -0.7:
                continue

            NewsArticle.objects.create(
                title=article["title"],
                description=article["description"],
                url=article["url"],
                sentiment=sentiment,
                polarity=polarity,
                published_at=parser.parse(article["publishedAt"]),
                source=article["source"],
                language=article.get("language", "en"),
            )
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")

    key_events = extract_key_events(news)
    return news, key_events
