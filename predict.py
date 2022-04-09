import csv, json, os, time, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "algo_trader.settings")
django.setup()

import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def setup_features_and_labels():
    data = pd.read_csv('./csv/dataset.csv', delimiter=',')
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
    model.add(Dense(8, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    print('----------\n', X_train.shape, y_train.shape)
    model.fit(X_train, y_train, epochs=8, batch_size=1, verbose=1)

if __name__ == "__main__":
    start_time = time.time()

    [labels, features, n_rows, n_cols] = setup_features_and_labels()
    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)
    setup_nn(X_train, y_train, n_rows)

    print("--- %ss prediction roundtrip ---" % round((time.time() - start_time), 1) )