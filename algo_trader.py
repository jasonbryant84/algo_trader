import sys, csv, json, os, time, argparse

from collect_data import build_datasets

# Handle flags/vars
# Example: python algo_trader.py --pair XRP/USDT --interval 5m --candles 50 (--cloudStorage)
parser = argparse.ArgumentParser(description="End-to-end algorithm")
parser.add_argument("--cloudStorage", help="store csvs in the cloud", action="store_true")
parser.add_argument( "--pair", dest="pair", default="BTC/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="50", help="number of candles for look back")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    datasets, wrote_file = build_datasets(
        pair=args.pair,
        interval=args.interval,
        candle_lookback_length=args.n_candles
    )

    write_success_str = "sucessefully wrote csv" if wrote_file else "failed to write csv"
    print(f"--- {round((time.time() - start_time), 1)}s Roundtrip and {write_success_str} ---")