import time

from datetime import datetime
from django.conf import settings

import pandas as pd
import numpy as np

from binance import Client
from binance.enums import HistoricalKlinesType
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from .taapi import TaapiInterface

class ExchangeHelper:
    def __init__(self):
        self.name = 'Exchange Helper'
        self.client = None

    class Meta:
        abstract = True

class BinanceHelper(ExchangeHelper):
    name = 'Binance Helper'
    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_SECRET)
    
    def __init__(self, pairs_of_interest, intervals_of_interest, klines_type='spot'):
        self.pairs_of_interest = pairs_of_interest
        self.intervals_of_interest = intervals_of_interest

        self.possible_intervals = [
            Client.KLINE_INTERVAL_1MINUTE,
            Client.KLINE_INTERVAL_3MINUTE,
            Client.KLINE_INTERVAL_5MINUTE,
            Client.KLINE_INTERVAL_15MINUTE,
            Client.KLINE_INTERVAL_30MINUTE,
            Client.KLINE_INTERVAL_30MINUTE,
            Client.KLINE_INTERVAL_1HOUR,
            Client.KLINE_INTERVAL_4HOUR,
            Client.KLINE_INTERVAL_1DAY,
            Client.KLINE_INTERVAL_1WEEK,
            Client.KLINE_INTERVAL_1MONTH
        ]
        self.klines_type = HistoricalKlinesType.SPOT if klines_type != 'futures' else HistoricalKlinesType.FUTURES

        self.datasets = {}
        self.full_dataset = None

    def convertIntervals(self, interval):
        if interval == '1m':
            return [Client.KLINE_INTERVAL_1MINUTE, '1 minute', '4 days ago UTC']
        if interval == '3m':
            return [Client.KLINE_INTERVAL_3MINUTE, '3 minutes', '6 days ago UTC']
        if interval == '5m':
            return [Client.KLINE_INTERVAL_5MINUTE, '5 minutes', '10 days ago UTC']
        elif interval == '15m':
            return [Client.KLINE_INTERVAL_15MINUTE, '15 minutes', '2 months ago UTC']
        elif interval == '30m':
            return [Client.KLINE_INTERVAL_30MINUTE, '30 minutes', '4 months ago UTC']
        elif interval == '1h':
            return [Client.KLINE_INTERVAL_1HOUR, '1 hour', '6 months ago UTC']
        elif interval == '4h':
            return [Client.KLINE_INTERVAL_4HOUR, '4 hours', '2 years ago UTC']
        elif interval == '1d':
            return [Client.KLINE_INTERVAL_1DAY, '1 day', '5 years ago UTC']
        elif interval == '1w':
            return [Client.KLINE_INTERVAL_1WEEK, '1 week', '9 years ago UTC']
        elif interval == '1M':
            return [Client.KLINE_INTERVAL_1MONTH, '1 month', '12 years ago UTC']
        else:
            return [Client.KLINE_INTERVAL_1DAY, '1 day (default)', '1 day ago UTC']

    def clean_data(self, klines):
        start_time = time.time()

        df = pd.DataFrame(
            klines,
            columns=['open_time_0', 'open_0', 'high_0', 'low_0', 'close_0', 'volume_0', 'close_time_0', 'quote_asset_vol_0', 'num_trades_0', 'taker_buy_base_asset_vol_0', 'taker_buy_quote_asset_vol_0', 'ignore_0'] # list of OHLCV
        )

        new_dtypes = {
            "open_0": np.float64,
            "high_0": np.float64,
            "low_0": np.float64,
            "close_0": np.float64,
            "volume_0": np.float64,
            "quote_asset_vol_0": np.float64,
            "taker_buy_base_asset_vol_0": np.float64,
            "taker_buy_quote_asset_vol_0": np.float64,
        }
        df = df.astype(new_dtypes)

        # Calculate Open/Close Difference
        df.insert(0, "diff_0", df["close_0"] - df["open_0"])

        # Up/Down
        df.insert(0, "was_up_0", df["diff_0"] > 0)
        df = df.astype({ "was_up_0": np.int8 })

        # Human readable datetime
        df.insert(0, "close_time_dt_0", pd.to_datetime(df["close_time_0"], unit='ms'))

        # Drop Open and Close times and Ignore column (maybe this is important info from Binance)
        del df["open_time_0"]
        del df["close_time_0"]
        del df["ignore_0"]

        # Reverse row order
        df = df.iloc[::-1]

        # taapi_interface = TaapiInterface()
        # request = taapi_interface.rsi(self.pair, interval) would need to pass interval into function

        print(f"--- {round((time.time() - start_time), 1)}s sseconds to clean data ---")
        return df


    def generate_klines(self, pair_str, interval):
        start_time = time.time()
        interval_data = self.convertIntervals(interval)

        klines = self.client.get_historical_klines(
            symbol=pair_str,
            interval=interval_data[0],
            start_str=interval_data[2],
            end_str=None,
            klines_type=self.klines_type
        )

        self.datasets[pair_str] = {"sets": []}

        print(f"--- {round((time.time() - start_time), 1)}s seconds to retrieve Binance data on {pair_str} with interval {interval} ---")
        return klines


    ############################################
    # Generating Dataset Logic                 #
    ############################################

    def generate_concatinated_columns_for_dataset(self, cleaned_data, candle_lookback_length):
        # Model features
        dataset = cleaned_data.copy()

        # Setup past related features by copying initial data and dropping unnecessary columns
        df_with_dropped_cols = cleaned_data.copy()
        del df_with_dropped_cols["close_time_dt_0"]
        del df_with_dropped_cols["was_up_0"]
        del df_with_dropped_cols["diff_0"]

        for i in range(1, candle_lookback_length):
            prevIStr = f"_{i-1}"
            prevSuffixOffset = -1 * len(prevIStr)

            # Update suffix to indicate past index
            df_with_dropped_cols = df_with_dropped_cols.rename(columns = lambda x : str(x)[:prevSuffixOffset]).add_suffix(f"_{i}")

            # Drop first row
            df_with_dropped_cols = df_with_dropped_cols.shift(periods=-1, axis="rows")

            # Concatenate
            updated_df = [dataset, df_with_dropped_cols]
            dataset = pd.concat(updated_df, axis=1)

        # Dropping the last "candle_lookback_length" rows as they will have NaN values due to shifting
        dataset = dataset.iloc[:(-1 * candle_lookback_length)]

        return dataset

    def generate_time_info_for_dataset(self, dataset, interval_of_interest):
        # Adding datetime info
        year = dataset["close_time_dt_0"].apply(lambda x: x.year)
        year.name = "year"
        month = dataset["close_time_dt_0"].apply(lambda x: x.month)
        month.name = "month"
        day = dataset["close_time_dt_0"].apply(lambda x: x.day)
        day.name = "day"
        hour = dataset["close_time_dt_0"].apply(lambda x: x.hour)
        hour.name = "hour"
        minute = dataset["close_time_dt_0"].apply(lambda x: x.minute)
        minute.name = "minute"
        day_of_week = dataset["close_time_dt_0"].apply(lambda x: x.weekday() + 1)
        day_of_week.name = "day_of_week"

        date_info_added_to_df = [dataset, year, month, day, hour, minute, day_of_week]
        dataset = pd.concat(date_info_added_to_df, axis=1)

        for interval in self.possible_intervals:
            interval_series_column = dataset["close_time_dt_0"].apply(lambda x: 1 if interval == interval_of_interest else 0)
            interval_series_column.name = f"is_{interval}"

            interval_added_to_df = [dataset, interval_series_column]
            dataset = pd.concat(interval_added_to_df, axis=1)

        return dataset
            

    def generate_pairs_dataset(self, dataset, curr_pair_of_interest):
        for pair in self.pairs_of_interest:
            pair_series_column = dataset["close_time_dt_0"].apply(lambda x: 1 if pair == curr_pair_of_interest else 0)
            pair_series_column.name = f"is_{pair.replace('/', '_')}"

            pair_added_to_df = [dataset, pair_series_column]
            dataset = pd.concat(pair_added_to_df, axis=1)

        return dataset

    def generate_datasets(self):    
        for pair in self.pairs_of_interest:
            sets = []

            for interval in self.intervals_of_interest:   
                pair = pair.replace('/', '')

                klines = self.generate_klines(pair, interval)
                cleaned_data = self.clean_data(klines)

                dataset = self.generate_concatinated_columns_for_dataset(cleaned_data, candle_lookback_length = 200)    
                dataset = self.generate_pairs_dataset(dataset, curr_pair_of_interest=pair)
                dataset = self.generate_time_info_for_dataset(dataset, interval)

                sets.append({
                    "interval": interval,
                    "dataset": dataset
                })


                temp_dataset = [self.full_dataset, dataset]
                self.full_dataset = pd.concat(temp_dataset, axis=0)

            self.datasets[pair]["sets"] = sets
        
        return [self.full_dataset, self.datasets]

    def __str__(self):
        return f"name: {self.name} klines_type: {self.klines_type}"