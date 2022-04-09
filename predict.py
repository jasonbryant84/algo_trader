# https://www.bmc.com/blogs/deep-learning-neural-network-tutorial-keras/

import sys, csv, json, os, time, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def setup_features_and_labels(filename):
    data = pd.read_csv(f"./csv/{filename}", delimiter=',')
    n_cols_in_data = data.shape[1]

    # skip over index, datetime, was_up (label - classification), diff(label - regression)
    features = data.iloc[:,4 : n_cols_in_data]
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

    print('X_train, X_test, y_train, y_test\n', X_train, '\n', X_test, '\n', y_train, '\n', y_test)

    # Normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]

def setup_nn(X_train, y_train, n_rows):
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))

    model.add(Dense(2*n_cols, activation='relu'))
    model.add(Dense(2*n_cols, activation='relu'))
    model.add(Dense(2*n_cols, activation='relu'))

    model.add(Dense(1, activation='sigmoid')) # Classification activation function
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1)

    return model

def predict(model, X_test, y_test):
    print('\n\n--- Predictions ---')
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"score: {score}\n")

if __name__ == "__main__":
    start_time = time.time()
        
    filename = "dataset.csv"
    if len(sys.argv) == 3:
        pair = sys.argv[1].replace('/','_')
        interval = sys.argv[2]
        filename = f"dataset_{pair}_{interval}.csv"

    [labels, features, n_rows, n_cols] = setup_features_and_labels(filename)
    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)
    
    layers = [n_cols]
    model = setup_nn(X_train, y_train, n_rows)
    predict(model, X_test, y_test)

    print("--- %ss prediction roundtrip ---" % round((time.time() - start_time), 1) )