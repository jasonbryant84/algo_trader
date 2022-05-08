# https://towardsdatascience.com/how-to-upload-and-download-files-from-aws-s3-using-python-2022-4c9b787b15f2

import time, argparse

from utils.collect_data_helpers import build_datasets

# Example: python collect_data.py --pair XRP/USDT --interval 5m --candles 10 --cloudStorage
parser = argparse.ArgumentParser(description="gather candlestick data from Binance and create datasets for machine learning")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="XRP/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="10", help="number of candles for look back")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    print('----- Collect Data -----')
    _, wrote_file = build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles,
        use_sub_intervals=True,
        cloudStorage=args.cloudStorage
    )

    write_success_str = "sucessefully wrote csv" if wrote_file else "failed to write csv"
    print(f"--- {round((time.time() - start_time), 1)}s Roundtrip and {write_success_str} ---")