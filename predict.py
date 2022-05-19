# https://www.bmc.com/blogs/deep-learning-neural-network-tutorial-keras/
# https://towardsdatascience.com/technical-analysis-of-stocks-using-ta-lib-305614165051


import sys, csv, json, glob, os, time, argparse

import tensorflow as tf
import pandas as pd
from keras import backend as K

import numpy as np

from utils.cloud_io import save_model

from utils.predict_helpers import setup_features_and_labels, setup_training_and_test_data, setup_nn, threshold_testing, predict

# Example: python predict.py --pair XRP/USDT --interval 5m --candles 10 --filename file.csv --epochs 1 --learning_rate 0.01 (--cloudStorage --loadLocalData --loadCloudModel --loadLocalModel)
parser = argparse.ArgumentParser(description="Predict: Generate neural network for buy/sell prediction")
parser.add_argument("--cloudStorage", help="store models in the cloud", action="store_true")
parser.add_argument("--noStorage", help="bypass local and cloud storage", action="store_true")
parser.add_argument("--loadLocalData", help="load a local csv file", action="store_true")
parser.add_argument("--loadLocalModel", help="load a local model file", action="store_true")
parser.add_argument("--loadCloudModel", help="load a GCP model file", action="store_true")
parser.add_argument("--liveMode", help="in live mode, thus test_size is 0", action="store_true")
parser.add_argument( "--pair", dest="pair", default="XRP/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="10", help="number of candles for look back")
parser.add_argument("--filename", dest="filename", default="no_file.csv", help="filename of csv to use for training")
parser.add_argument("--epochs", dest="n_epochs", default="35", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.01", help="learning rate to be used to train the neural network")
args = parser.parse_args()

bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]

if __name__ == "__main__":
    start_time = time.time()
    print('----- Predict -----')

    interval = args.interval
    candles = args.n_candles

    pair = args.pair.replace("/", "_")
    filename_dataset = args.filename
    filename_model = None
    print('======================', filename_dataset, args.filename)

    labels, features, n_rows, n_cols = setup_features_and_labels(
        pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        filename=args.filename, # getting filename.csv only
        loadLocalData=args.loadLocalData, # will load from cloud (GCP)
        noStorage=args.noStorage,
        dataset=None # only used if noStorage is true (not storing csv)
    )

    [X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test] = setup_training_and_test_data(labels, features)

    model = None
    # TODO implement properly
    if args.loadLocalModel:
        path_model = f"./models/{pair}/*"
        list_of_files = glob.glob(path_model)
        latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    elif args.loadCloudModel:
        pair_sans_slash = pair.replace("/", "_")

        storage_client = storage.Client()
        PREFIX=f"models/{pair_sans_slash}/{interval}/{candles}_candles/"
        blobs = list(storage_client.list_blobs(bucket_name, prefix=PREFIX, fields="items(name)"))
        blob_names = [
            blob_name.name[len(PREFIX):] 
            for blob_name 
            in blobs 
            if blob_name.name[-1] == "/" and blob_name.name[-2].isdigit()
        ]

        model_filename = blob_names[-1][:-1]
        latest_filepath = f"gs://{bucket_name}/{PREFIX}{model_filename}" 

        # path_model = f"./models/{pair}/*"
        # list_of_files = glob.glob(path_model)
        # latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    else:
        filename_model = args.filename.replace("dataset_", "").replace(".csv", "")

        model, _ = setup_nn(
            y_train,
            n_cols,
            X_train=X_train_scaled,
            n_epochs=int(args.n_epochs),
            learning_rate=args.learning_rate
        )

    predict(model, y_test, X_test=X_test_scaled)
    
    save_model(
        pair,
        interval,
        candles,
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