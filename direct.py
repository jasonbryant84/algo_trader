import json, os, time, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

from django.conf import settings
from django.http import HttpResponse
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from interface.utils.helpers import BinanceHelper

BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
BINANCE_SECRET = os.environ['BINANCE_SECRET']


# Be careful for unfinished candles when making prediction ie most recent row is a currently active candlestick

def no_request():
    start_time = time.time()

    pairs_of_interest = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT", "LUNA/USDT", "AVAX/USDT", "DOT/USDT", "DOGE/USDT", "MATIC/USDT", "LTC/USDT", "TRX/USDT"]
    pairs_of_interest = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT"]

    intervals_of_interest=["5m", "15m", "4h"]

    helper = BinanceHelper(
        pairs_of_interest=pairs_of_interest,
        intervals_of_interest=intervals_of_interest,
    )
    datasets = helper.generate_datasets()

    print(datasets)
    print("--- %ss Roundtrip ---" % round((time.time() - start_time), 1) )

if __name__ == "__main__":
    no_request()