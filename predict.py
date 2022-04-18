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

# Handle flags/vars
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


def setup_features_and_labels(pair, interval, candle_lookback_length, filename, loadLocalData):
    os.environ['GOOGLE_APPLICAITON_CREDENTIALS'] = "credentials.json"

    try:
        filename_local = f"./datasets/{pair}/{interval}/{candle_lookback_length}_candles/{filename}" 
        filename_gcp = f"gs://algo-trader-staging/datasets/{pair}/{interval}/{candle_lookback_length}_candles/{filename}"
        
        filename = filename_local if loadLocalData else filename_gcp
        print('filename', filename)
        data = pd.read_csv(filename)
        print('data', data)

        n_cols_in_data = data.shape[1]

        # skip over index, datetime, was_up (label - classification), diff(label - regression)
        features = data.iloc[:,3 : n_cols_in_data]
        features = features[:-1]

        labels = data["was_up_0"]
        labels = labels.shift(periods=-1, axis="rows")
        labels = labels[:-1]

        n_rows = features.shape[0]
        n_cols = features.shape[1]

        return labels, features, n_rows, n_cols

    except Exception as e:
        print(e)
        print('Unable to load csv')
        exit()
        return False

def setup_training_and_test_data(labels, features):
    X = features
    y = np.ravel(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]

def setup_nn(X_train, y_train, n_rows, n_epochs = 3, learning_rate = 0.01):
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))

    model.add(Dense(n_cols, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(n_cols, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(n_cols, activation='relu'))
    # model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid')) # Classification activation function
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    K.set_value(model.optimizer.learning_rate, learning_rate)
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=1, verbose=1)

    return model

def threshold_testing(y_test, y_pred, thresholds):
    # NOTA BENE: Limit false positives (fp)! ie false buy signals
    tests = []

    n_positives = np.count_nonzero(y_test == 1)
    n_negatives = np.count_nonzero(y_test == 0)

    max_accuracy = { "value": 0 }
    lowest_fp = { "value": y_test.shape[0] }

    for threshold in thresholds:
        temp_pred = y_pred
        temp_pred = np.where(temp_pred > threshold, 1, 0)

        tn, fp, fn, tp = confusion_matrix(y_test, temp_pred).ravel()
        n_predictions = tn + fp + fn + tp # could just be rows but whatever for now
        n_correct = tn + tp
        accuracy = n_correct / n_predictions

        if accuracy > max_accuracy["value"]:
            max_accuracy["value"] = accuracy
            max_accuracy["threshold"] = threshold

        if fp < lowest_fp["value"]:
            lowest_fp["value"] = fp
            lowest_fp["threshold"] = threshold

        obj = {
            "threshold": threshold,
            "accuracy": accuracy,
            "tn": tn, 
            "fp": fp,
            "fn": fn, 
            "tp": tp
        }

        tests.append(obj)

    return tests, max_accuracy, lowest_fp, n_positives, n_negatives

def predict(model, X_test, y_test):
    print(f"\n--- Keras Evaluate - Predictions (threshold = 0.5 by default) ---") # for evaluate method below
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"score: {score}\n")

    print('-----Jason \n\n', X_test.shape)
    y_pred = model.predict(X_test).flatten()
    confusion_matrices = threshold_testing(
        y_test,
        y_pred,
        thresholds = np.arange(start=0.5, stop=0.75, step=0.01)
    )

    pprint(confusion_matrices)

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
    if args.loadLocalModel:
        path_model = f"./models/{pair}/*"
        list_of_files = glob.glob(path_model)
        latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    elif args.loadCloudModel:
        path_model = f"./models/{pair}/*"
        list_of_files = glob.glob(path_model)
        latest_filepath = max(list_of_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_filepath)
    else:
        model = setup_nn(
            X_train,
            y_train,
            n_rows,
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