import time
from utils.helpers import BinanceHelper
from utils.cloud_io import write_dataset_csvs

def build_datasets(pair, interval, candle_lookback_length, cloudStorage):
    # Alter the following 2 arrays if desired
    # Entering a pair and interval in command line will take precedent
    pairs_of_interest = [pair] if pair else ["XRP/USDT"]
    intervals_of_interest = [interval] if interval else ["5m"]
    candle_lookback_length = candle_lookback_length or 50

    helper = BinanceHelper(
        pairs_of_interest=pairs_of_interest,
        intervals_of_interest=intervals_of_interest,
        candle_lookback_length=candle_lookback_length
    )

    print('\nGenerating dataset(s)...')
    start_time = time.time()
    [full_dataset, datasets] = helper.generate_datasets()
    print(f"--- {round((time.time() - start_time), 1)}s to generate dataset ---\n")

    wrote_file = write_dataset_csvs(datasets, helper, cloudStorage)

    return datasets, wrote_file