# https://towardsdatascience.com/how-to-upload-and-download-files-from-aws-s3-using-python-2022-4c9b787b15f2

import sys, csv, json, os, time, argparse

import s3fs
import pandas as pd

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager # TODO readup on those other two imports
from utils.helpers import BinanceHelper
from google.cloud import storage

# Handle flags/vars
# Example: python collect_data.py --pair XRP/USDT --interval 5m --candles 50 (--cloudStorage)
parser = argparse.ArgumentParser(description="gather candlestick data from Binance and create datasets for machine learning")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

def write_csvs(datasets, interfaceHelper):
    # Setting credentials for Google Cloud Platform (Cloud Storage specifically)
    os.environ['GOOGLE_APPLICAITON_CREDENTIALS'] = "credentials.json"

    try:
        storage_client = storage.Client()
        buecket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]
        bucket = storage_client.get_bucket(buecket_name)

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
                month = month if month > 9 else f"0{month}"
                day = dataset.iloc[0]["day"]
                day = day if day > 9 else f"0{day}"
                year = dataset.iloc[0]["year"]
                hour = dataset.iloc[0]["hour"]
                hour = hour if hour > 9 else f"0{hour}"
                minute = dataset.iloc[0]["minute"]
                minute = minute if minute > 9 else f"0{minute}"
                print('dataset\n', dataset)
                
                # Local Storage
                if not args.cloudStorage:
                    filename = f"{path}/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"

                    dataset.to_csv(filename)
                    print(f"---- wrote (locally) file {filename}")
                
                # GCP - Cloud Storage
                else:
                    filename_gcp = f"datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"
                    
                    bucket.blob(filename_gcp).upload_from_string(dataset.to_csv(), 'text/csv')
                    print(f"---- wrote (in GCP) file {filename_gcp}")

                return True

    except Exception as e:
        print(e)
        return False

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

    wrote_file = write_csvs(datasets, helper)
    write_success_str = "sucessefully wrote csv" if wrote_file else "failed to write csv"

    print(f"--- {round((time.time() - start_time), 1)}s Roundtrip and {write_success_str} ---")

if __name__ == "__main__":
    build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles
    )