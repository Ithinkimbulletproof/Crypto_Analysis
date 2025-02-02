import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from django.db import transaction
from django.db.models import Max
from crypto_analysis.models import NewsArticle
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CD_API_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"


def get_last_published_date():
    last_date = NewsArticle.objects.aggregate(max_date=Max("published_at"))["max_date"]
    if last_date:
        logger.info(f"Последняя опубликованная новость в БД: {last_date}")
    else:
        logger.info("В БД ещё нет новостей.")
    return last_date or datetime.now(timezone.utc) - timedelta(days=2000)


def fetch_news_from_rss(url, source_name):
    articles = []
    min_date = get_last_published_date()

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for item in root.findall(".//item"):
            title = (
                item.find("title").text
                if item.find("title") is not None
                else "No Title"
            )
            url = item.find("link").text if item.find("link") is not None else ""
            pub_date_text = (
                item.find("pubDate").text if item.find("pubDate") is not None else None
            )

            if pub_date_text:
                pub_date = datetime.strptime(pub_date_text, "%a, %d %b %Y %H:%M:%S %z")
            else:
                pub_date = datetime.now(timezone.utc)

            if pub_date > min_date:
                articles.append(
                    {
                        "title": title[:255],
                        "url": url,
                        "published_at": pub_date,
                        "source": source_name,
                    }
                )

    except Exception as e:
        logger.error(f"Ошибка запроса к {source_name}: {str(e)}")
        return None

    logger.info(f"{source_name}: получено {len(articles)} новостей")
    return sorted(articles, key=lambda x: x["published_at"], reverse=True)


def fetch_news():
    return fetch_news_from_rss(CD_API_URL, "CoinDesk")


@transaction.atomic
def save_articles_to_db(articles):
    new_count = 0
    for article in articles:
        try:
            _, created = NewsArticle.objects.get_or_create(
                url=article["url"],
                defaults={
                    "title": article["title"],
                    "published_at": article["published_at"],
                    "source": article["source"],
                },
            )
            if created:
                new_count += 1
        except Exception as e:
            logger.error(f"Ошибка сохранения {article['url']}: {str(e)}")
    logger.info(
        f"Сохранено {new_count} новых статей, пропущено {len(articles) - new_count} дубликатов"
    )


def run_full_import():
    try:
        articles = fetch_news()
        if articles:
            save_articles_to_db(articles)
            logger.info(f"Загружено {len(articles)} новостей")
        else:
            logger.info("Нет новых новостей для сохранения.")
    except Exception as e:
        logger.error(f"Ошибка импорта: {str(e)}")


if __name__ == "__main__":
    run_full_import()
