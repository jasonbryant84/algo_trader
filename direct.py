import json, os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

from django.conf import settings
from django.http import HttpResponse
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from interface.utils.helpers import BinanceHelper

BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
BINANCE_SECRET = os.environ['BINANCE_SECRET']

def hello():
    pair = "XRP/USDT"
    parin_sans_slash = pair.replace('/','')
    interval = "1d"

    helper = BinanceHelper(pair, interval)
    helper.generate_klines()
    helper.clean_data()
    helper.generate_features()

if __name__ == "__main__":
    hello()