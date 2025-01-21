import os
import time
import ccxt
import logging
from dotenv import load_dotenv
from django.db.models import Max
from datetime import datetime, timezone
from crypto_analysis.models import MarketData


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_start_date():
    start_date = datetime(2021, 1, 1)
    return int(start_date.timestamp() * 1000)


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


def fetch_data():
    exchanges = [ccxt.binance(), ccxt.kraken()]
    symbols = os.getenv("CRYPTOPAIRS").split(",")
    timeframe = "1h"
    default_start_date = get_default_start_date()
    logger.info(f"Получение данных для криптовалютных пар: {symbols}")
    logger.info(f"Стартовый timestamp для данных: {default_start_date}")
    for exchange in exchanges:
        logger.info(f"Начало обработки биржи: {exchange.id}")
        try:
            process_exchange(exchange, symbols, default_start_date, timeframe)
            logger.info(f"Завершена обработка биржи: {exchange.id}")
        except Exception as e:
            logger.error(f"Ошибка работы с биржей {exchange.id}: {str(e)}")
    logger.info("Загрузка данных завершена")


def process_exchange(exchange, symbols, default_start_date, timeframe):
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
                fetch_and_store_data(exchange, symbol, since, timeframe)
            else:
                logger.warning(f"Пара {symbol} не найдена на {exchange.id}")
    except Exception as e:
        logger.error(f"Ошибка при обработке биржи {exchange.id}: {str(e)}")


def fetch_and_store_data(exchange, symbol, since, timeframe):
    retries = 3
    retry_delay = 60
    all_data = []
    while retries > 0:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since)
            if not data:
                logger.info(f"Данные завершены для {symbol} на {exchange.id}")
                break
            logger.info(f"Получено {len(data)} записей для {symbol} на {exchange.id}")
            for record in data:
                timestamp = record[0]
                naive_date = datetime.utcfromtimestamp(timestamp / 1000)
                aware_date = naive_date.replace(tzinfo=timezone.utc)
                logger.debug(
                    f"Timestamp: {timestamp}, Naive Date: {naive_date}, Aware Date: {aware_date}"
                )
                all_data.append(
                    (
                        symbol,
                        exchange.id,
                        timestamp,
                        record[1],
                        record[2],
                        record[3],
                        record[4],
                        record[5],
                    )
                )
            since = data[-1][0] + 1
        except ccxt.NetworkError as e:
            retries -= 1
            logger.warning(f"Ошибка сети: {str(e)}. Осталось попыток: {retries}")
            if retries == 0:
                logger.error(f"Попытки исчерпаны для {symbol} на {exchange.id}")
            else:
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")
            break
    store_data_bulk(all_data)
    logger.info(
        f"Общее количество загруженных записей для {symbol} на {exchange.id}: {len(all_data)}"
    )


def store_data_bulk(all_data):
    if not all_data:
        return

    records_to_check = {
        (symbol, datetime.fromtimestamp(timestamp / 1000, timezone.utc), exchange_id)
        for symbol, exchange_id, timestamp, _, _, _, _, _ in all_data
    }

    existing_records = set(
        MarketData.objects.filter(
            cryptocurrency__in={symbol for symbol, _, _, _, _, _, _, _ in all_data},
            date__in={
                datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                for _, _, timestamp, _, _, _, _, _ in all_data
            },
            exchange__in={exchange_id for _, exchange_id, _, _, _, _, _, _ in all_data},
        ).values_list("cryptocurrency", "date", "exchange")
    )

    market_data_objects = []
    for (
        symbol,
        exchange_id,
        timestamp,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
    ) in all_data:
        date = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        if (symbol, date, exchange_id) not in existing_records:
            market_data_objects.append(
                MarketData(
                    cryptocurrency=symbol,
                    date=date,
                    exchange=exchange_id,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                )
            )

    if market_data_objects:
        try:
            MarketData.objects.bulk_create(market_data_objects, batch_size=1000)
            logger.info(
                f"Данные успешно сохранены для {len(market_data_objects)} записей"
            )
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных в базу: {str(e)}")
    else:
        logger.info("Нет новых данных для сохранения")
