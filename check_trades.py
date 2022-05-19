import argparse, time
import numpy as np
import matplotlib.pyplot as plt

from utils.cloud_io import fetch_predictions, fetch_latest_candles, write_prediction_csv
from utils.check_trades_helpers import check_trades
from utils.formatting import bcolors

# Example: python check_trades.py --pair XRP/USDT --interval 5m --candles 10 --cloudStorage
parser = argparse.ArgumentParser(description="Update and clean predictions table")
parser.add_argument("--cloudStorage", help="get predcitions from cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="XRP/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="10", help="number of candles for look back")
args = parser.parse_args()

if __name__ == "__main__":
    print(f"{bcolors.OKCYAN}----- Check Trades -----{bcolors.ENDC}")

    predictions_df = check_trades(
        pair=args.pair,
        interval=args.interval,
        n_candles=args.n_candles,
        cloudStorage=args.cloudStorage
    )

    window = 60
    predictions_df = predictions_df.head(window)
    reference_start_price = predictions_df.iloc[window-1]["close"]

    # Accuracy
    predictions_array = predictions_df["prediction_correct"].value_counts()
    correct = predictions_array[1] if 1 in predictions_array else 0
    incorrect = predictions_array[0] if 0 in predictions_array else 0
    accuracy = correct / (correct + incorrect)
    print(f"accuracy: {(accuracy * 100).round(decimals = 2)}%")

    correct_buys_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 1) & (predictions_df['actual'] == 1), 'diff'].sum()
    correct_sells_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 1) & (predictions_df['actual'] == 0), 'diff'].sum()
    incorrect_buys_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 0) & (predictions_df['actual'] == 1), 'diff'].sum()
    incorrect_sells_sum = predictions_df.loc[(predictions_df['prediction_correct'] == 0) & (predictions_df['actual'] == 0), 'diff'].sum()
    
    correct_delta = correct_buys_sum + abs(correct_sells_sum)
    incorrect_delta = incorrect_buys_sum + abs(incorrect_sells_sum)
    overall_delta = correct_delta - incorrect_delta

    print('correct_delta', correct_delta.round(decimals = 3), ":", correct_buys_sum.round(decimals = 3), correct_sells_sum.round(decimals = 3))
    print('incorrect_delta', incorrect_delta.round(decimals = 3), ":", incorrect_buys_sum.round(decimals = 3), incorrect_sells_sum.round(decimals = 3))
    print('overall_delta', overall_delta.round(decimals = 3))

    print(f"delta percentage {((overall_delta / reference_start_price) * 100).round(decimals = 3)}%")
    reference = ( ((predictions_df.iloc[0]["close"] - reference_start_price) / reference_start_price) * 100).round(decimals = 3)
    print(f"ref delta percentage {reference}% (without trading) ")

    print(predictions_df)

    closes = predictions_df["close"]

    graph, (plot1, plot2) = plt.subplots(1, 2)

    plot1.set_title(f"Predictions for ({args.pair})")
    plot1.plot(closes, color='r', label=f"Closes: length - {len(closes)}")
    # plot1.plot(X_test_close, color='g', label=f"X_test_close: length - {len(X_test_close)}")
    plot1.invert_xaxis()
    plot1.legend()

    # plot2.set_title("Testing Data")
    # plot2.plot(X_test["close_0"], color='g', label=f"X_test_close: length - {len(X_test_close)}")
    # plot2.plot(X_test["open_0"], color='b', label=f"X_test_open: length - {len(X_test_close)}")
    # plot2.invert_xaxis()
    # plot2.legend()
    
    graph.tight_layout()
    plt.show(block=False)
