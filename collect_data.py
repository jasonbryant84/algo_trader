# https://towardsdatascience.com/how-to-upload-and-download-files-from-aws-s3-using-python-2022-4c9b787b15f2

import sys, csv, json, os, time, argparse

import s3fs
import pandas as pd

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager # TODO readup on those other two imports
from utils.helpers import BinanceHelper
from utils.cloud_io import write_csvs
from google.cloud import storage

# Handle flags/vars
# Example: python collect_data.py --pair XRP/USDT --interval 5m --candles 50 (--cloudStorage)
parser = argparse.ArgumentParser(description="gather candlestick data from Binance and create datasets for machine learning")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

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

    wrote_file = write_csvs(datasets, helper, cloudStorage=args.cloudStorage)
    write_success_str = "sucessefully wrote csv" if wrote_file else "failed to write csv"

    print(f"--- {round((time.time() - start_time), 1)}s Roundtrip and {write_success_str} ---")

if __name__ == "__main__":
    build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles
    )