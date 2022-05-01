import sys, os

from pprint import pprint

from keras.models import Sequential
import pandas as pd
from keras.layers import Dense, Dropout
from keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]

def setup_features_and_labels(pair, interval, candle_lookback_length, filename, loadLocalData, noStorage, dataset):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google-credentials.json"

    try:
        data = None

        if not noStorage:
            filename_local = f"./datasets/{pair}/{interval}/{candle_lookback_length}_candles/{filename}" 
            filename_gcp = f"gs://{bucket_name}/datasets/{pair}/{interval}/{candle_lookback_length}_candles/{filename}"
            
            filename = filename_local if loadLocalData else filename_gcp
            data = pd.read_csv(filename)
        else:
            data = dataset

        n_cols_in_data = data.shape[1]

        # skip over index, datetime, was_up (label - classification), diff(label - regression)
        features = data.iloc[:,2 : n_cols_in_data] if noStorage else data.iloc[:,3 : n_cols_in_data]
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

def setup_nn(X_train, y_train, n_cols, n_epochs = 3, learning_rate = 0.01):
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))

    model.add(Dense(n_cols, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_cols, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_cols, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid')) # Classification activation function
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    K.set_value(model.optimizer.learning_rate, learning_rate)

    # print('---------- Sanity -----------')
    # import pdb
    # pdb.set_trace()
    # print(X_train.shape(), y_train.shape(), n_cols, n_epochs)
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

    # y_pred = model.predict(X_test).flatten()
    # confusion_matrices = threshold_testing(
    #     y_test,
    #     y_pred,
    #     thresholds = np.arange(start=0.5, stop=0.75, step=0.01)
    # )

    # pprint(confusion_matrices)