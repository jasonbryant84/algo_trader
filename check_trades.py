import argparse, time
import numpy as np

from utils.cloud_io import fetch_predictions, fetch_latest_candles, write_prediction_csv
from utils.check_trades_helpers import check_trades

# Example: python check_trades.py --pair XRP/USDT --interval 5m --candles 10 --cloudStorage
parser = argparse.ArgumentParser(description="Update and clean predictions table")
parser.add_argument("--cloudStorage", help="get predcitions from cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

if __name__ == "__main__":
    predictions_df = check_trades(
        pair=args.pair,
        interval=args.interval,
        n_candles=args.n_candles,
        cloudStorage=args.cloudStorage
    )
    
    print('predictions_df.shape', predictions_df.shape)
    predictions_df = predictions_df.head(100)

    # Accuracy
    predictions_array = predictions_df["prediction_correct"].value_counts()
    correct = predictions_array[1] 
    incorrect = predictions_array[0]
    accuracy = correct / (correct + incorrect)
    print(f"accuracy: {accuracy}")

    correct_buys_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 1) & (predictions_df['actual'] == 1), 'diff'].sum()
    correct_sells_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 1) & (predictions_df['actual'] == 0), 'diff'].sum()
    incorrect_buys_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 0) & (predictions_df['actual'] == 1), 'diff'].sum()
    incorrect_sells_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 0) & (predictions_df['actual'] == 0), 'diff'].sum()
    
    correct_delta = correct_buys_sum + abs(correct_sells_sum)
    incorrect_delta = incorrect_buys_sum + abs(incorrect_sells_sum)
    overall_delta = correct_delta - incorrect_delta

    print('correct_delta', correct_delta, ":", correct_buys_sum, correct_sells_sum)
    print('incorrect_delta', incorrect_delta, ":", incorrect_buys_sum, incorrect_sells_sum)
    print('overall_delta', overall_delta)
