import argparse, time
import numpy as np

from utils.cloud_io import fetch_predictions, fetch_latest_candles, write_prediction_csv

# Example: python check_trades.py --pair XRP/USDT --interval 5m --candles 10 --cloudStorage
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
        
    if "buy_sell_actual" in predictions_df.keys():
        del predictions_df["buy_sell_actual"]   
    if "actual" in predictions_df.keys():
        del predictions_df["actual"]
    
    # Get latest candles from exchange
    latest_candles = fetch_latest_candles(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles
    )

    latest_candles = latest_candles[['was_up_0', 'diff_0', 'close_0', 'open_0', 'high_0', 'low_0', 'close_time_dt_0', ]]

    latest_candles.rename(columns={'was_up_0':'actual'}, inplace=True)
    latest_candles.rename(columns={'close_time_dt_0':'closing_time'}, inplace=True)
    latest_candles.rename(columns={'diff_0':'diff'}, inplace=True)
    latest_candles.rename(columns={'close_0':'close'}, inplace=True)
    latest_candles.rename(columns={'open_0':'open'}, inplace=True)
    latest_candles.rename(columns={'high_0':'high'}, inplace=True)
    latest_candles.rename(columns={'low_0':'low'}, inplace=True)
    
    predictions_df = predictions_df.astype({ "closing_time": np.datetime64 })

    # predictions_df.update(latest_candles)
    # Join on closing_time
    predictions_df = predictions_df.merge(latest_candles, on='closing_time', how='left')

    # import pdb
    # pdb.set_trace()
    if "close_x" in predictions_df.keys():
        # del predictions_df["buy_sell_actual_x"]
        del predictions_df["diff_x"]
        del predictions_df["close_x"]
        del predictions_df["open_x"]
        del predictions_df["high_x"]
        del predictions_df["low_x"]

    # import pdb
    # pdb.set_trace()
    if "close_y" in predictions_df.keys():
        # predictions_df.rename(columns={'buy_sell_actual_y':'buy_sell_actual'}, inplace=True)
        predictions_df.rename(columns={'diff_y':'diff'}, inplace=True)
        predictions_df.rename(columns={'close_y':'close'}, inplace=True)
        predictions_df.rename(columns={'open_y':'open'}, inplace=True)
        predictions_df.rename(columns={'high_y':'high'}, inplace=True)
        predictions_df.rename(columns={'low_y':'low'}, inplace=True)

    predictions_df["prediction_correct"] = (predictions_df["actual"] == predictions_df["buy_sell_prediction"]).astype(int)
    predictions_df.insert(1, 'actual', predictions_df.pop('actual'))

    predictions_df['diff'].round(decimals = 5)
    
    # Write to cloud storage
    write_prediction_csv(predictions_df, predictions_filename, path)