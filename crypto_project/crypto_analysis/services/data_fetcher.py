import time
import ccxt
import logging
import statistics
from django.db.models import Max
from datetime import datetime, timezone
from crypto_analysis.models import MarketData

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
    return int(last_date.timestamp() * 1000) if last_date else default_start_date


def fetch_data(symbols):
    exchanges = [ccxt.binance(), ccxt.kraken(), ccxt.okx()]
    timeframe = "1h"
    default_start_date = get_default_start_date()
    logger.info(f"[{datetime.now()}] Получение данных для криптовалютных пар: {symbols}")
    logger.info(f"[{datetime.now()}] Стартовый timestamp для данных: {default_start_date}")

    for exchange in exchanges:
        logger.info(f"[{datetime.now()}] Начало обработки биржи: {exchange.id}")
        try:
            process_exchange_for_all_symbols(
                exchange, symbols, default_start_date, timeframe
            )
            logger.info(f"[{datetime.now()}] Завершена обработка биржи: {exchange.id}")
        except Exception as e:
            logger.error(f"[{datetime.now()}] Ошибка работы с биржей {exchange.id}: {str(e)}")
    logger.info(f"[{datetime.now()}] Загрузка данных завершена")


def process_exchange_for_all_symbols(exchange, symbols, default_start_date, timeframe):
    try:
        logger.info(f"[{datetime.now()}] Загрузка рынков для биржи {exchange.id}")
        exchange.load_markets()
        logger.info(f"[{datetime.now()}] Рынки успешно загружены для биржи {exchange.id}")
        for symbol in symbols:
            logger.info(f"[{datetime.now()}] Обработка пары {symbol} на {exchange.id}")
            if symbol in exchange.markets:
                last_date = get_last_date(symbol, exchange)
                since = get_since(last_date, default_start_date)
                logger.info(f"[{datetime.now()}] Начинаю загрузку данных для {symbol} с {since}")
                fetch_and_store_data(exchange, symbol, since, timeframe)
            else:
                logger.warning(f"[{datetime.now()}] Пара {symbol} не найдена на {exchange.id}")
    except Exception as e:
        logger.error(f"[{datetime.now()}] Ошибка при обработке биржи {exchange.id}: {str(e)}")


def fetch_and_store_data(exchange, symbol, since, timeframe):
    retries = 3
    retry_delay = 60
    all_data = []
    logger.info(f"[{datetime.now()}] Начало загрузки данных для {symbol} на {exchange.id}")
    while retries > 0:
        try:
            logger.debug(f"[{datetime.now()}] Запрос данных для {symbol} с {since} на {exchange.id}")
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since)
            if not data:
                logger.info(f"[{datetime.now()}] Нет данных для {symbol} на {exchange.id}. Завершаем.")
                break
            logger.debug(f"[{datetime.now()}] Загружено {len(data)} записей для {symbol} на {exchange.id}")
            for record in data:
                timestamp = record[0]
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
            logger.warning(f"[{datetime.now()}] Ошибка сети: {str(e)}. Осталось попыток: {retries}")
            if retries == 0:
                logger.error(f"[{datetime.now()}] Попытки исчерпаны для {symbol} на {exchange.id}")
            else:
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"[{datetime.now()}] Ошибка загрузки: {str(e)} для {symbol} на {exchange.id}")
            break
    logger.info(f"[{datetime.now()}] Сохранение данных для {symbol} на {exchange.id}")
    store_data_bulk(all_data, exchange.id)
    logger.info(f"[{datetime.now()}] Данные успешно сохранены для {symbol} на {exchange.id}")


def store_data_bulk(all_data, exchange_id):
    if not all_data:
        logger.info(f"[{datetime.now()}] Нет данных для сохранения для {exchange_id}")
        return

    data_by_symbol_date = {}

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

        if (symbol, date) not in data_by_symbol_date:
            data_by_symbol_date[(symbol, date)] = {
                "open_prices": [],
                "high_prices": [],
                "low_prices": [],
                "close_prices": [],
                "volumes": [],
                "exchange_ids": set(),
            }

        data_by_symbol_date[(symbol, date)]["open_prices"].append(open_price)
        data_by_symbol_date[(symbol, date)]["high_prices"].append(high_price)
        data_by_symbol_date[(symbol, date)]["low_prices"].append(low_price)
        data_by_symbol_date[(symbol, date)]["close_prices"].append(close_price)
        data_by_symbol_date[(symbol, date)]["volumes"].append(volume)
        data_by_symbol_date[(symbol, date)]["exchange_ids"].add(exchange_id)

    logger.info(f"[{datetime.now()}] Начало сохранения данных в базу для {exchange_id}")

    market_data_objects = []
    for (symbol, date), data in data_by_symbol_date.items():
        avg_open_price = statistics.mean(data["open_prices"])
        avg_high_price = statistics.mean(data["high_prices"])
        avg_low_price = statistics.mean(data["low_prices"])
        avg_close_price = statistics.mean(data["close_prices"])
        avg_volume = statistics.mean(data["volumes"])

        exchange_ids = ", ".join(data["exchange_ids"])

        existing_record = MarketData.objects.filter(
            cryptocurrency=symbol, date=date, exchange=exchange_ids
        ).first()

        if existing_record:
            existing_record.open_price = avg_open_price
            existing_record.high_price = avg_high_price
            existing_record.low_price = avg_low_price
            existing_record.close_price = avg_close_price
            existing_record.volume = avg_volume
            existing_record.save()
            logger.debug(f"[{datetime.now()}] Обновление записи для {symbol} на {date}")
        else:
            market_data_objects.append(
                MarketData(
                    cryptocurrency=symbol,
                    date=date,
                    exchange=exchange_ids,
                    open_price=avg_open_price,
                    high_price=avg_high_price,
                    low_price=avg_low_price,
                    close_price=avg_close_price,
                    volume=avg_volume,
                )
            )

    if market_data_objects:
        logger.debug(f"[{datetime.now()}] Добавление {len(market_data_objects)} новых записей")
        MarketData.objects.bulk_create(market_data_objects, batch_size=1000)

    logger.info(f"[{datetime.now()}] Завершено сохранение данных для {exchange_id}")
