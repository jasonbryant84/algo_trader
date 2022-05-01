import sys, csv, json, os, time, argparse, datetime

from utils.collect_data_helpers import build_datasets
from predict import setup_features_and_labels, setup_training_and_test_data, setup_nn, predict
from utils.cloud_io import save_model

# Example: python setup.py --pair XRP/USDT --interval 5m --candles 50 --epochs 1 --learning_rate 0.03  --cloudStorage
parser = argparse.ArgumentParser(description="End-to-end algorithm")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument("--noStorage", help="get predcitions from cloud", action="store_true")
parser.add_argument("--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
parser.add_argument("--epochs", dest="n_epochs", default="1", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.03", help="learning rate to be used to train the neural network")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    print('----- Setup -----')

    # Collect Data
    ##############################################
    datasets, wrote_file = build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
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
    # import pdb
    # pdb.set_trace()
    # print(dataset.iloc[0][:1])

    # Predict
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
        dataset=dataset # only used if noStorage is true (not storing csv)
    )
    print('labels\n', labels)
    print('features\n', features)
    print('n_rows\n', n_rows)
    print('n_cols\n', n_cols)

    [X_train, X_test, y_train, y_test] = setup_training_and_test_data(labels, features)

    model = setup_nn(
        X_train,
        y_train,
        n_cols,
        n_epochs=int(args.n_epochs),
        learning_rate=args.learning_rate
    )

    predict(model, X_test, y_test)
    
    filename_model = filename.replace("dataset_", "").replace(".csv", "") if not args.noStorage else filename_model
    success = save_model(
        pair,
        filename_model,
        model,
        cloudStorage=True
    )

    print(f"======== model written: {filename_model} - success {success}")
    print(f"\n\n\n===== {round((time.time() - start_time), 1)}s to execute setup.py =====")