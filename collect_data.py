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

def write_csvs(datasets, interfaceHelper):
    for pair in datasets:
        dataset_obj = datasets[pair]
        
        for curr_set in dataset_obj["sets"]:
            dataset = curr_set["dataset"]
            interval = curr_set["interval"]
            path = f"./datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles"

            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            month = dataset.iloc[0]["month"]
            day = dataset.iloc[0]["day"]
            year = dataset.iloc[0]["year"]
            hour = dataset.iloc[0]["hour"]
            minute = dataset.iloc[0]["minute"]

            print('dataset\n', dataset)
            filename = f"{path}/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"

            dataset.to_csv(filename)
            print(f"---- wrote file {filename}")

def build_datasets(pair, interval, candle_lookback_length):
    start_time = time.time()

    # Alter the following 2 arrays if desired
    # Entering a pair and interval in command line will take precedent
    pairs_of_interest = [pair] if pair else ["XRP/USDT"]
    intervals_of_interest = [interval] if interval else ["5m"]
    candle_lookback_length = candle_lookback_length or 50

    helper = BinanceHelper(
        pairs_of_interest=pairs_of_interest,
        intervals_of_interest=intervals_of_interest,
        candle_lookback_length=candle_lookback_length
    )

    [full_dataset, datasets] = helper.generate_datasets()

    write_csvs(datasets, helper)

    print("--- %ss Roundtrip ---" % round((time.time() - start_time), 1) )

if __name__ == "__main__":
    # TODO: add prompts if there are no parameters passed
    # python collect_data.py XRP/USDT 5m 50 

    pair = None
    interval = None
    candle_lookback_length = None

    if len(sys.argv) > 1:
        pair = sys.argv[1]
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    if len(sys.argv) > 3:
        candle_lookback_length = sys.argv[3]

    build_datasets(pair, interval, candle_lookback_length)