import time
from utils.helpers import BinanceHelper
from utils.cloud_io import write_dataset_csvs

def build_datasets(pair, interval, candle_lookback_length, use_sub_intervals, cloudStorage, noStorage):
    # Alter the following 2 arrays if desired
    # Entering a pair and interval in command line will take precedent
    pairs_of_interest = [pair] if pair else ["XRP/USDT"]
    intervals_of_interest = [interval] if interval else ["5m"]
    candle_lookback_length = candle_lookback_length or 10

    helper = BinanceHelper(
        pairs_of_interest=pairs_of_interest,
        intervals_of_interest=intervals_of_interest,
        candle_lookback_length=candle_lookback_length
    )

    print('\nGenerating dataset(s)...')
    start_time = time.time()
    full_dataset, datasets, sub_interval, sub_interval_unit = helper.generate_datasets(use_sub_intervals, use_for_prediction=True)
    print(f"--- {round((time.time() - start_time), 1)}s to generate dataset ---\n")

    print('noStorage', noStorage)
    if not noStorage:
        wrote_file = write_dataset_csvs(datasets, helper, cloudStorage)
        return datasets, wrote_file, sub_interval, sub_interval_unit

    return datasets, None, sub_interval, sub_interval_unit