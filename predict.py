# https://www.bmc.com/blogs/deep-learning-neural-network-tutorial-keras/
# https://towardsdatascience.com/technical-analysis-of-stocks-using-ta-lib-305614165051


import sys, csv, json, os, time, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

from pprint import pprint

import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def setup_features_and_labels(filename):
    data = pd.read_csv(f"./csv/{filename}", delimiter=',')
    n_cols_in_data = data.shape[1]

    # skip over index, datetime, was_up (label - classification), diff(label - regression)
    features = data.iloc[:,3 : n_cols_in_data]
    features = features[:-1]

    labels = data["was_up_0"]
    labels = labels.shift(periods=-1, axis="rows")
    labels = labels[:-1]

    n_rows = features.shape[0]
    n_cols = features.shape[1]

    return [labels, features, n_rows, n_cols]

def setup_training_and_test_data(labels, features):
    X = features
    y = np.ravel(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]

def setup_nn(X_train, y_train, n_rows, n_epochs=3):
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))

    model.add(Dense(n_cols, activation='relu'))
    model.add(Dense(n_cols, activation='relu'))
    model.add(Dense(n_cols, activation='relu'))

    model.add(Dense(1, activation='sigmoid')) # Classification activation function
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=1, verbose=1)

    return model

def threshold_testing(y_test, y_pred, thresholds):
    # NOTA BENE: Limit false positives (fp)! ie false buy signals
    tests = []

    for threshold in thresholds:
        temp_pred = y_pred
        temp_pred = np.where(temp_pred > threshold, 1, 0)

        tn, fp, fn, tp = confusion_matrix(y_test, temp_pred).ravel()
        n_predictions = tn + fp + fn + tp # could just be rows but whatever for now
        n_correct = tn + tp

        obj = {
            "threshold": threshold,
            "accuracy": n_correct / n_predictions,
            "tn": tn, 
            "fp": fp,
            "fn": fn, 
            "tp": tp
        }

        tests.append(obj)

    return tests

def predict(model, X_test, y_test):
    print(f"\n--- Keras Evaluate - Predictions (threshold = 0.5 by default) ---") # for evaluate method below
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"score: {score}\n")

    y_pred = model.predict(X_test).flatten()
    confusion_matrices = threshold_testing(
        y_test,
        y_pred,
        thresholds = np.arange(start=0.5, stop=0.9, step=0.01)
    )

    pprint(confusion_matrices)
    # print(confusion_matrices)

if __name__ == "__main__":
    start_time = time.time()
        
    # Default parameter values if none supplied on command line
    pair = None
    interval = None
    n_epochs = 1

    # Default derived value (below)
    filename = "dataset.csv"

    if len(sys.argv) >= 3:
        pair = sys.argv[1].replace('/','_')
        interval = sys.argv[2]
        filename = f"dataset_{pair}_{interval}.csv"
    if len(sys.argv) >= 4:
        n_epochs = int(sys.argv[3])
    
    [labels, features, n_rows, n_cols] = setup_features_and_labels(filename)
    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)
    
    layers = [n_cols]
    model = setup_nn(X_train, y_train, n_rows, n_epochs)
    predict(model, X_test, y_test)

    print("--- %ss prediction roundtrip ---" % round((time.time() - start_time), 1) )