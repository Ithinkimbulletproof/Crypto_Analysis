import os
import asyncio
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import ccxt
import ccxt.async_support as ccxt_async
import logging
from dotenv import load_dotenv
from django.db.models import Max
from crypto_analysis.models import MarketData
from datetime import datetime, timezone, timedelta
from asgiref.sync import sync_to_async
from django.db import close_old_connections

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_start_date():
    start_date = datetime.now() - timedelta(days=1095)
    return int(start_date.timestamp() * 1000)


@sync_to_async
def get_last_date(symbol, exchange_id):
    try:
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


async def fetch_data():
    exchanges = [ccxt_async.binance(), ccxt_async.kraken()]
    symbols = os.getenv("CRYPTOPAIRS").split(",")
    timeframe = "1h"
    default_start_date = get_default_start_date()

    logger.info(f"Загрузка данных для пар: {symbols}")
    await asyncio.gather(
        *[
            process_exchange(exchange, symbols, default_start_date, timeframe)
            for exchange in exchanges
        ]
    )
    logger.info("Загрузка данных завершена")


async def process_exchange(exchange, symbols, default_start_date, timeframe):
    logger.info(f"Обработка биржи: {exchange.id}")
    try:
        await exchange.load_markets()
        exchange_id = exchange.id
        tasks = [
            fetch_and_store_data(
                exchange,
                symbol,
                get_since(await get_last_date(symbol, exchange_id), default_start_date),
                timeframe,
            )
            for symbol in symbols
            if symbol in exchange.markets
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Ошибка на бирже {exchange.id}: {str(e)}")
    finally:
        await exchange.close()


async def fetch_and_store_data(exchange, symbol, since, timeframe):
    retries, all_data = 3, []
    while retries > 0:
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe, since=since)
            if not data:
                logger.info(f"Нет новых данных для {symbol}")
                break
            logger.info(f"Получено {len(data)} записей для {symbol}")
            all_data.extend([(symbol, exchange.id, *record) for record in data])
            since = data[-1][0] + 1
            retries = 3
        except ccxt.NetworkError as e:
            retries -= 1
            logger.warning(f"Ошибка сети: {e}. Попыток: {retries}")
            await asyncio.sleep(60 * (4 - retries))
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            break
    await store_data_bulk(all_data)


@sync_to_async
def store_data_bulk(all_data):
    close_old_connections()
    if not all_data:
        return

    existing = set(
        MarketData.objects.filter(
            cryptocurrency__in=[d[0] for d in all_data],
            date__in=[
                datetime.fromtimestamp(d[2] / 1000, tz=timezone.utc) for d in all_data
            ],
            exchange__in=[d[1] for d in all_data],
        ).values_list("cryptocurrency", "date", "exchange")
    )

    new_data = [
        MarketData(
            cryptocurrency=symbol,
            exchange=exchange_id,
            date=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            open_price=open,
            high_price=high,
            low_price=low,
            close_price=close,
            volume=volume,
        )
        for (symbol, exchange_id, timestamp, open, high, low, close, volume) in all_data
        if (
            symbol,
            datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            exchange_id,
        )
        not in existing
    ]

    if new_data:
        MarketData.objects.bulk_create(new_data, batch_size=1000)
        logger.info(f"Сохранено {len(new_data)} записей")
    else:
        logger.info("Нет новых данных")


if __name__ == "__main__":
    asyncio.run(fetch_data())
