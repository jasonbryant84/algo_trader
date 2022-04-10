import sys, csv, json, os, time, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

from django.conf import settings
from django.http import HttpResponse
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from interface.utils.helpers import BinanceHelper

BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
BINANCE_SECRET = os.environ['BINANCE_SECRET']

# Be careful for unfinished candles when making prediction ie most recent row is a currently active candlestick

def build_datasets(pair, interval):
    start_time = time.time()

    # Alter the following 2 arrays if desired
    # Entering a pair and interval in command line will take precedent
    pairs_of_interest = [pair] if pair else ["BTC/USDT"]
    intervals_of_interest = [interval] if interval else ["5m"]

    helper = BinanceHelper(
        pairs_of_interest=pairs_of_interest,
        intervals_of_interest=intervals_of_interest,
    )
    [full_dataset, datasets] = helper.generate_datasets()

    pairs_array = [pair.replace('/', '_') for pair in pairs_of_interest]
    full_dataset.to_csv(
        f"./csv/dataset_{'_'.join(pairs_array)}_{'_'.join(intervals_of_interest)}.csv"
    )

    print('full_dataset', full_dataset)
    print("--- %ss Roundtrip ---" % round((time.time() - start_time), 1) )

if __name__ == "__main__":
    # TODO: add prompts if there are no parameters passed
    pair = None
    interval = None

    if len(sys.argv) == 2:
        pair = sys.argv[1]
    elif len(sys.argv) == 3:
        pair = sys.argv[1]
        interval = sys.argv[2]

    build_datasets(pair, interval)