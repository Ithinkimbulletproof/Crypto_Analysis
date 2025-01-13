import ccxt
import logging
from datetime import datetime
from django.db.models import Max
from django.utils import timezone
from crypto_analysis.models import MarketData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_start_date():
    start_date = datetime(2021, 1, 1)
    return int(start_date.timestamp() * 1000)


def fetch_data():
    exchanges = [ccxt.binance()]
    symbols = ["BTC/USDT"]
    timeframe = "1h"
    default_start_date = get_default_start_date()
    logger.info(f"Получение данных для криптовалютных пар: {symbols}")
    logger.info(f"Стартовый timestamp для данных: {default_start_date}")
    for exchange in exchanges:
        logger.info(f"Начало обработки биржи: {exchange.id}")
        try:
            process_exchange(exchange, symbols, default_start_date)
            logger.info(f"Завершена обработка биржи: {exchange.id}")
        except Exception as e:
            logger.error(f"Ошибка работы с биржей {exchange.id}: {str(e)}")
    logger.info("Загрузка данных завершена")


def process_exchange(exchange, symbols, default_start_date):
    logger.info(f"Начало обработки биржи: {exchange.id}")
    try:
        exchange.load_markets()
        for symbol in symbols:
            logger.info(f"Обработка пары {symbol} на {exchange.id}")
            if symbol in exchange.markets:
                last_date = get_last_date(symbol, exchange)
                logger.info(
                    f"Последняя дата для {symbol} на {exchange.id}: {last_date}"
                )
                since = get_since(last_date, default_start_date)
                logger.info(f"Дата начала для загрузки данных: {since}")
                fetch_and_store_data(exchange, symbol, since)
            else:
                logger.warning(f"Пара {symbol} не найдена на {exchange.id}")
    except Exception as e:
        logger.error(f"Ошибка при обработке биржи {exchange.id}: {str(e)}")


def fetch_and_store_data(exchange, symbol, since):
    retries = 3
    all_data = []
    while retries > 0:
        try:
            data = exchange.fetch_ohlcv(symbol, "1h", since=since)
            if not data:
                logger.info(f"Данные завершены для {symbol} на {exchange.id}")
                break
            logger.info(f"Получено {len(data)} записей для {symbol} на {exchange.id}")
            for record in data:
                timestamp = record[0]
                naive_date = datetime.utcfromtimestamp(timestamp / 1000)
                aware_date = timezone.make_aware(naive_date)
                logger.info(
                    f"Timestamp: {timestamp}, Naive Date: {naive_date}, Aware Date: {aware_date}"
                )
                all_data.append(record)
            since = data[-1][0] + 1
        except ccxt.NetworkError as e:
            retries -= 1
            logger.warning(f"Ошибка сети: {str(e)}. Повторные попытки: {retries}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")
            break
    for record in all_data:
        store_data(symbol, exchange, record)
    logger.info(
        f"Общее количество загруженных записей для {symbol} на {exchange.id}: {len(all_data)}"
    )


def store_data(symbol, exchange, record):
    timestamp = record[0]
    naive_date = datetime.utcfromtimestamp(timestamp / 1000)
    aware_date = timezone.make_aware(naive_date)
    logger.info(
        f"Storing data with timestamp: {timestamp}, Naive Date: {naive_date}, Aware Date: {aware_date}"
    )
    try:
        MarketData.objects.update_or_create(
            cryptocurrency=symbol,
            date=aware_date,
            exchange=exchange.id if hasattr(exchange, "id") else exchange,
            defaults={
                "open_price": record[1],
                "high_price": record[2],
                "low_price": record[3],
                "close_price": record[4],
                "volume": record[5],
            },
        )
        logger.info(f"Данные для {symbol} на {exchange.id} успешно сохранены")
    except Exception as e:
        logger.error(f"Ошибка добавления записи: {str(e)}")


def get_last_date(symbol, exchange):
    try:
        exchange_id = getattr(exchange, "id", None)
        if not exchange_id:
            raise AttributeError("Объект exchange не имеет атрибута 'id'.")
        last_date = MarketData.objects.filter(
            cryptocurrency=symbol, exchange=exchange_id
        ).aggregate(Max("date"))["date__max"]
        logger.info(f"Последняя дата для {symbol} на {exchange_id}: {last_date}")
        return last_date
    except Exception as e:
        logger.error(f"Ошибка при получении последней даты: {str(e)}")
        return None


def get_since(last_date, default_start_date):
    if last_date:
        since = int(last_date.timestamp() * 1000)
    else:
        since = default_start_date
    return since
