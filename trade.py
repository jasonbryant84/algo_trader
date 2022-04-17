import sys, csv, json, glob, os, time

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

def get_data(pair, interval, candle_lookback_length):
    helper = BinanceHelper(
        pairs_of_interest=[pair],
        intervals_of_interest=[interval],
        candle_lookback_length=candle_lookback_length
    )

    [_, datasets] = helper.generate_datasets()

    return datasets

if __name__ == "__main__":
    # TODO: add prompts if there are no parameters passed
    # >>> python trade.py XRP/USDT 15m 50

    start_time = time.time()

    pair = sys.argv[1]
    pair_sans_slash = pair.replace("/", "_")
    interval = sys.argv[2]
    candle_lookback_length = sys.argv[3]

    # maybe leverage dataset stored (csv or db/sql) or just generate since we gotta fetch anyway
    data = get_data(pair, interval, candle_lookback_length)
    first_row = data[pair_sans_slash]["sets"][0]["dataset"].iloc[0]

    print('\n\n','first_row', first_row, '\n\n')
    closing_time = first_row["close_time_dt_0"]
    del first_row["close_time_dt_0"]
    del first_row["was_up_0"]

    # make a prediction
    path_model = f"./models/{pair_sans_slash}/*"
    list_of_files = glob.glob(path_model)
    latest_filepath = max(list_of_files, key=os.path.getctime)
    model = tf.keras.models.load_model(latest_filepath)

    buy_sell_array = model.predict(first_row)
    buy_sell = buy_sell_array[0][0]
    buy_sell_str = "Buy" if buy_sell == 1 else "Sell"
    # show the inputs and predicted outputs


    ############################
    ############################
    ############################
    ############################
    ############################
    ############################
    # STOP USING MOST RECENT FILE AND USE REGEX TO SYNC WITH TIME INTERVAL YOU DINGUS

    print(f"----- Used model: {latest_filepath} -----")
    print(f"----- Predicted trade {buy_sell_str} (previous close time {closing_time}) {buy_sell_array} -----")

    print(f"--- {round((time.time() - start_time), 1)}s trade roundtrip (pair: {pair}, interval: {interval}) ---")