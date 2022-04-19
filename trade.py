import sys, csv, json, glob, os, time, argparse

from pprint import pprint

import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense, Dropout
from keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from utils.helpers import BinanceHelper

from google.cloud import storage

# Handle flags/vars
# Example: python trade.py --pair XRP/USDT --interval 5m --candles 50 (--loadCloudModel --cloudStorage --loadLocalModel)
parser = argparse.ArgumentParser(description="Generate neural network for buy/sell prediction")
parser.add_argument("--cloudStorage", help="store models in the cloud", action="store_true")
# parser.add_argument("--loadLocalData", help="load a local csv file", action="store_true")
# parser.add_argument("--loadLocalModel", help="load a local model file", action="store_true")
parser.add_argument("--loadCloudModel", help="load a GCP model file", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

os.environ['GOOGLE_APPLICAITON_CREDENTIALS'] = "credentials.json"
bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]

def get_data(pair, interval, candle_lookback_length):
    helper = BinanceHelper(
        pairs_of_interest=[pair],
        intervals_of_interest=[interval],
        candle_lookback_length=candle_lookback_length
    )

    [_, datasets] = helper.generate_datasets()

    return datasets

if __name__ == "__main__":
    start_time = time.time()

    pair = args.pair #sys.argv[1]
    pair_sans_slash = pair.replace("/", "_")
    interval = args.interval #sys.argv[2]
    candle_lookback_length = args.n_candles #sys.argv[3]

    # maybe leverage dataset stored (csv or db/sql) or just generate since we gotta fetch anyway
    data = get_data(pair, interval, candle_lookback_length)
    first_row = data[pair_sans_slash]["sets"][0]["dataset"].iloc[:1] # []:1] is Dataframe [0] is Series

    print('\n\n','first_row', first_row, '\n\n')
    closing_time = first_row["close_time_dt_0"]
    del first_row["close_time_dt_0"]
    del first_row["was_up_0"]

    # make a prediction
    list_of_files = None
    latest_filepath = None

    if args.loadCloudModel:
        storage_client = storage.Client()
        PREFIX=f"models/{pair_sans_slash}"
        blobs = list(storage_client.list_blobs(bucket_name, prefix=PREFIX, fields="items(name)"))
        blob_names = [
            blob_name.name[len(PREFIX):] 
            for blob_name 
            in blobs 
            if blob_name.name[-1] == "/" and blob_name.name[-2].isdigit()
        ]

        filename = blob_names[-1][1:]
        latest_filepath = f"gs://{bucket_name}/models/{pair_sans_slash}/{filename}" 
    else:
        path_model_local = f"./models/{pair_sans_slash}/*"
        list_of_files = glob.glob(path_model_local)
        latest_filepath = max(list_of_files, key=os.path.getctime)

    model = tf.keras.models.load_model(latest_filepath)

    buy_sell_array = model.predict(first_row)
    buy_sell = buy_sell_array[0][0]
    buy_sell_str = "Buy" if buy_sell == 1 else "Sell"

    print(f"----- Used model: {latest_filepath} -----")
    print(f"----- Predicted trade {buy_sell_str} (previous close time {closing_time}) {buy_sell_array}) -----")

    print(f"--- {round((time.time() - start_time), 1)}s trade roundtrip (pair: {pair}, interval: {interval}) ---")