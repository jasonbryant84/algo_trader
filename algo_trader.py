import sys, csv, json, os, time, argparse

from utils.collect_data_helpers import build_datasets
from predict import setup_features_and_labels, setup_training_and_test_data, setup_nn, predict
from utils.cloud_io import save_model

# Handle flags/vars
# Example: python algo_trader.py --pair XRP/USDT --interval 5m --candles 50 --epochs 1 --learning_rate 0.03  --cloudStorage
parser = argparse.ArgumentParser(description="End-to-end algorithm")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
parser.add_argument("--epochs", dest="n_epochs", default="1", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.03", help="learning rate to be used to train the neural network")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    # Collect Data
    ##############################################
    datasets, wrote_file = build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        cloudStorage=args.cloudStorage
    )

    print('wrote_file', wrote_file)
    write_success_str = "sucessefully wrote csv" if wrote_file else "failed to write csv"
    print(f"--- {round((time.time() - start_time), 1)}s Roundtrip and {write_success_str} ---")


    # Predict
    ##############################################
    pair = args.pair.replace("/", "_")
    filename = wrote_file.split("/")[-1]

    labels, features, n_rows, n_cols = setup_features_and_labels(
        pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        filename=filename, # getting filename.csv only
        loadLocalData=False # will load from cloud (GCP)
    )

    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)

    model = setup_nn(
        X_train,
        y_train,
        n_cols,
        n_epochs=int(args.n_epochs),
        learning_rate=args.learning_rate
    )

    predict(model, X_test, y_test)
    
    filename_model = filename.replace("dataset_", "").replace(".csv", "")
    save_model(
        pair,
        filename_model,
        model,
        cloudStorage=True
    )