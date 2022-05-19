import sys, csv, json, os, time, argparse, datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from google.cloud import storage
from utils.collect_data_helpers import build_datasets
from predict import setup_features_and_labels, setup_training_and_test_data, setup_nn, predict
from utils.cloud_io import save_model
from utils.formatting import bcolors

# Example: python setup.py --pair XRP/USDT --interval 5m --candles 10 --epochs 33 --learning_rate 0.01 --cloudStorage
# Example: python setup.py --pair XRP/USDT --interval 5m --candles 10 --epochs 33 --learning_rate 0.01  --noStorage (--liveMode)
parser = argparse.ArgumentParser(description="End-to-end algorithm")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument("--noStorage", help="get predcitions from cloud", action="store_true")
parser.add_argument("--loadCloudModel", help="load a GCP model file", action="store_true")
parser.add_argument("--liveMode", help="in live mode, thus test_size is 0", action="store_true")
parser.add_argument("--pair", dest="pair", default="XRP/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="10", help="number of candles for look back")
parser.add_argument("--epochs", dest="n_epochs", default="35", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.01", help="learning rate to be used to train the neural network")
args = parser.parse_args()

bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]

if __name__ == "__main__":
    start_time = time.time()
    # https://stackoverflow.com/questions/25351968/how-can-i-display-full-non-truncated-dataframe-information-in-html-when-conver
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 2000)
    # pd.set_option('display.float_format', '{:20,.2f}'.format)
    # pd.set_option('display.max_colwidth', None)

    print(f"{bcolors.OKCYAN}----- Setup -----{bcolors.ENDC}", datetime.datetime.utcnow())

    interval = args.interval
    interval_num = int(interval[:-1])
    candles = args.n_candles
    use_sub_intervals=True

    # Collect Data
    ##############################################
    datasets, wrote_file, sub_interval, _ = build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        use_sub_intervals=use_sub_intervals,
        cloudStorage=args.cloudStorage,
        noStorage=args.noStorage
    )

    # Use dataset and bypass csv implementation
    dataset = None
    filename_model = None
    if wrote_file is None:
        now = datetime.datetime.utcnow()
        dataset = datasets[args.pair.replace("/","_")]["sets"][0]["dataset"]

        # Generate filename_model
        month = now.month if now.month > 9 else f"0{now.month}"
        day = now.day if now.day > 9 else f"0{now.day}"
        year = now.year
        hour = now.hour if now.hour > 9 else f"0{now.hour}"
        minute = now.minute if now.minute > 9 else f"0{now.minute}"

        filename_model = f"{args.pair.replace('/','_')}_{args.interval}_{args.n_candles}candles_{month}-{day}-{year}_{hour}-{minute}"

        print('**************************** SHAPE: ', dataset.shape)

    # Predict w/ Model
    ##############################################
    pair = args.pair.replace("/", "_")
    filename = wrote_file.split("/")[-1] if wrote_file is not None else None

    labels, features, n_rows, n_cols = setup_features_and_labels(
        pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        filename=filename, # getting filename.csv only
        loadLocalData=False, # will load from cloud (GCP)
        noStorage=args.noStorage,
        dataset=dataset, # only used if noStorage is true (not storing csv)
        interval_num=interval_num
    )
    print('labels\n', labels)
    print('features\n', features)
    print('n_rows\n', n_rows)
    print('n_cols\n', n_cols)
    print('dataset\n', dataset)

    [X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test] = setup_training_and_test_data(
        labels,
        features,
        live_mode=args.liveMode
    )

    # Model
    ##############################################
    model = None
    sample_weight = None

    if args.loadCloudModel:
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

        model = tf.keras.models.load_model(latest_filepath)

        model.fit(
            X_train_scaled, # X_train_scaled no? and what are the dimensions w/ and w/o loading existing mode
            y_train, # y_train no? and what are the dimensions w/ and w/o loading existing mode
            epochs=int(args.n_epochs),
            batch_size=1,
            verbose=1
        )
    else:
        model, sample_weight = setup_nn(
            X_train=X_train_scaled,
            y_train=y_train,
            n_cols=n_cols,
            n_epochs=int(args.n_epochs),
            learning_rate=args.learning_rate
        )

    score, y_test_predictions = predict(model, X_test_scaled, y_test)
    
    if not args.liveMode:
        X_train_close = X_train["close_0"]
        X_test_close = X_test["close_0"]
        predictions_results = np.where(y_test == y_test_predictions, 1, 0)
        incorrect_predictions = np.where(predictions_results == 0, 1, 0) * X_test_close


        graph, (plot1, plot2) = plt.subplots(1, 2)

        plot1.set_title(f"Temporal Training Split for ({args.pair}) / Test accuracy {score}%")
        plot1.plot(X_train_close, color='r', label=f"X_train_close: length - {len(X_train_close)}")
        plot1.plot(X_test_close, color='g', label=f"X_test_close: length - {len(X_test_close)}")
        plot1.invert_xaxis()
        plot1.legend()

        plot2.set_title(f"Testing Data: use_sub_intervals is {use_sub_intervals}")
        plot2.plot(X_test["close_0"], color='c', label=f"X_test_close: length - {len(X_test_close)}")
        plot2.plot(X_test["open_0"], color='b', label=f"X_test_open: length - {len(X_test_close)}")

        for i in range(len(incorrect_predictions)):
            if incorrect_predictions[i] != 0:
                plot2.plot(i, incorrect_predictions[i],'ro')

        plot2.invert_xaxis()
        plot2.legend()

        # plot3.set_title(f"Predictions: {len(predictions_results)} length")
        # # plot3.plot(predictions_results, color='y', label=f"Predictions: length - {len(predictions_results)}")
        # plot3.plot(incorrect_predictions, color='m', label=f"Incorrect Predictions - {len(incorrect_predictions)}")
        # plot3.invert_xaxis()
        # plot3.legend()

        # plot3.set_title("Sample weighting")
        # plot3.plot(sample_weight, color='g', label=f"Sample Weight: length - {len(sample_weight)}")
        # plot3.invert_xaxis()
        # plot3.legend()
        
        graph.tight_layout()
        plt.show()

    filename_model = filename.replace("dataset_", "").replace(".csv", "") if not args.noStorage else filename_model
    success = save_model(
        pair,
        interval,
        candles,
        filename_model,
        model,
        cloudStorage=True
    )

    print(f"======== model written: {filename_model} - success {success}")
    print(f"\n\n\n===== {round((time.time() - start_time), 1)}s to execute setup.py =====")