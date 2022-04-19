# https://www.bmc.com/blogs/deep-learning-neural-network-tutorial-keras/
# https://towardsdatascience.com/technical-analysis-of-stocks-using-ta-lib-305614165051


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

from utils.cloud_io import save_model

from utils.predict_helpers import setup_features_and_labels, setup_training_and_test_data, setup_nn, threshold_testing, predict

# Example: python predict.py --pair XRP/USDT --interval 5m --candles 50 --filename file.csv --epochs 1 --learning_rate 0.03 (--cloudStorage --loadLocalData --loadLocalModel)
parser = argparse.ArgumentParser(description="Generate neural network for buy/sell prediction")
parser.add_argument("--cloudStorage", help="store models in the cloud", action="store_true")
parser.add_argument("--loadLocalData", help="load a local csv file", action="store_true")
parser.add_argument("--loadLocalModel", help="load a local model file", action="store_true")
parser.add_argument("--loadCloudModel", help="load a GCP model file", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
parser.add_argument("--filename", dest="filename", default="no_file.csv", help="filename of csv to use for training")
parser.add_argument("--epochs", dest="n_epochs", default="1", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.03", help="learning rate to be used to train the neural network")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    pair = args.pair.replace("/", "_")
    filename_dataset = args.filename
    filename_model = args.filename.replace("dataset_", "").replace(".csv", "")
    print('======================', filename_dataset, args.filename)

    labels, features, n_rows, n_cols = setup_features_and_labels(
        pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        filename=args.filename,
        loadLocalData=args.loadLocalData
    )

    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)
    
    model = None
    # TODO implement properly
    if args.loadLocalModel:
        path_model = f"./models/{pair}/*"
        list_of_files = glob.glob(path_model)
        latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    # TODO implement properly
    elif args.loadCloudModel:
        path_model = f"./models/{pair}/*"
        list_of_files = glob.glob(path_model)
        latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    else:
        model = setup_nn(
            X_train,
            y_train,
            n_cols,
            n_epochs=int(args.n_epochs),
            learning_rate=args.learning_rate
        )

    predict(model, X_test, y_test)
    
    save_model(
        pair,
        filename_model,
        model,
        cloudStorage=args.cloudStorage
    )


        # if reconstructed_model:
        #     new_model = reconstructed_model
        #     print(f"--- Retrieve model {latest_filepath} ---")

        #     labels, features, n_rows, n_cols = setup_features_and_labels(pair, interval, candle_lookback_length, latest_dataset)
        #     [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)
        #     new_model.fit(X_train, y_train, epochs=n_epochs, batch_size=1, verbose=1)

        #     # Let's check:
        #     np.testing.assert_allclose(
        #         new_model.predict(X_test), reconstructed_model.predict(X_test)
        #     )

        #     predict(new_model, X_test, y_test)
        #     save_model(pair, filename_model, new_model)

    print(f"--- {round((time.time() - start_time), 1)}s prediction roundtrip (pair: {pair}) ---")