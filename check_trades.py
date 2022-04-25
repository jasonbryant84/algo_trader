import argparse, time
import numpy as np

from utils.cloud_io import fetch_predictions, fetch_latest_candles, write_prediction_csv

# Example: python check_trades.py --pair XRP/USDT --interval 5m --candles 50 --cloudStorage
parser = argparse.ArgumentParser(description="Update and clean predictions table")
parser.add_argument("--cloudStorage", help="get predcitions from cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    # Setup variables
    pair = args.pair
    pair_sans_slash = pair.replace("/", "_")
    path = f"{pair_sans_slash}/{args.interval}/{args.n_candles}_candles"
    predictions_filename = f"predictions_{path.replace('/', '_')}.csv"

    # Retrieve predictions csv and convert to dataframe
    predictions_df = fetch_predictions(predictions_filename, path)
    
    # Drop duplicates and keep the oldest record (the first prediction made)
    predictions_df = predictions_df.drop_duplicates(subset='closing_time', keep="last")
    
    # Get latest candles from exchange
    latest_candles = fetch_latest_candles(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles
    )
    latest_candles = latest_candles[['was_up_0', 'close_time_dt_0']] # lets add open/close later

    latest_candles.rename(columns={'was_up_0':'buy_sell_actual'}, inplace=True)
    latest_candles.rename(columns={'close_time_dt_0':'closing_time'}, inplace=True)

    predictions_df = predictions_df.astype({ "closing_time": np.datetime64 })

    predictions_df = predictions_df.set_index('closing_time')
    latest_candles = latest_candles.set_index('closing_time')

    predictions_df.update(latest_candles)

    predictions_df["prediction_correct"] = (predictions_df["buy_sell_actual"] == predictions_df["buy_sell_prediction"]).astype(int)

    predictions_df.reset_index(inplace=True)

    # Write to cloud storage
    write_prediction_csv(predictions_df, predictions_filename, path)