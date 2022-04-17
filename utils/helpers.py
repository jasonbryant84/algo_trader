import os, time, math

from datetime import datetime

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from binance import Client
from binance.enums import HistoricalKlinesType
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

import talib as ta
# from .taapi import TaapiInterface

class ExchangeHelper:
    def __init__(self):
        self.name = 'Exchange Helper'
        self.client = None

    class Meta:
        abstract = True


class BinanceHelper(ExchangeHelper):
    BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
    BINANCE_SECRET = os.environ['BINANCE_SECRET']

    name = 'Binance Helper'
    client = Client(BINANCE_API_KEY, BINANCE_SECRET)
    
    def __init__(self, pairs_of_interest, intervals_of_interest, candle_lookback_length, klines_type='spot'):
        self.pairs_of_interest = pairs_of_interest
        self.intervals_of_interest = intervals_of_interest
        self.candle_lookback_length = int(candle_lookback_length)

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
        interval_unit = interval[-1]
        interval_num = int(interval[:-1])
        indicators_cushion = self.candle_lookback_length * 500 # cushion is in "interval_units"
        binance_cushion = 1 # api goes by datetime so adding one extra day/week/month cushion

        if interval_unit == "m":
            minutes_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion
            hours_total = int(math.ceil(minutes_total / 60)) + 1 
            days_total = int(math.ceil(minutes_total / 1440)) + binance_cushion
            return [interval, f"{interval_num} minutes", f"{days_total} days ago UTC"]
        
        elif interval_unit == "h":
            hours_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion
            days_total = int(math.ceil(hours_total / 24)) + binance_cushion
            return [interval, f"{interval_num} hours", f"{days_total} days ago UTC"]

        elif interval_unit == "d":
            days_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return [interval, f"{interval_num} days", f"{days_total} days ago UTC"]
        
        elif interval_unit == "w":
            weeks_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return [interval, f"{interval_num} weeks", f"{weeks_total} weeks ago UTC"]
        
        elif interval_unit == "M":
            months_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return [interval, f"{interval_num} months", f"{months_total} months ago UTC"]

    def generate_indicators_for_dataset(self, df):
        df.insert(0, "ADX_20_0", ta.ADX(df['high_0'], df['low_0'], df['close_0'], timeperiod=20))
        df.insert(0, "SMA_5_0", ta.SMA(df['close_0'], 5))
        df.insert(0, "SMA_20_0", ta.SMA(df['close_0'], 20))
        df.insert(0, "EMA_20_0", ta.EMA(df['close_0'], 20))
        df.insert(0, "rsi_5_0", ta.RSI(df['close_0'], 5))
        df.insert(0, "rsi_14_0", ta.RSI(df['close_0'], 14))

        dif, dea, bar = ta.MACD(df['close_0'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df.insert(0, "macd_dif_0", dif)
        df.insert(0, "macd_dea_0", dea)
        df.insert(0, "macd_bar_0", bar)
        
        up, mid, low = ta.BBANDS(df['close_0'], timeperiod = 20)
        df.insert(0, "low_band_20_0", low)
        df.insert(0, "mid_band_20_0", mid)
        df.insert(0, "up_band_20_0", up)

        # Visualize
        # df[['close_0', 'SMA_5_0', 'EMA_20_0', 'up_band_20_0', 'mid_band_20_0', 'low_band_20_0']].head(100).plot(figsize=(24,12))
        # plt.show()

        # df[['macd_dif_0', 'macd_dea_0', 'macd_bar_0']].head(100).plot(figsize=(24,12))
        # plt.show()
        
        return df.dropna()

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

        df = self.generate_indicators_for_dataset(df)

        # Up/Down Label
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

        print(f"--- {round((time.time() - start_time), 1)}s sseconds to clean data ---")
        return df


    def generate_klines(self, pair_str, interval, pair_as_key):
        start_time = time.time()
        interval_data = self.convertIntervals(interval)

        klines = self.client.get_historical_klines(
            symbol=pair_str,
            interval=interval_data[0],
            start_str=interval_data[2],
            end_str=None,
            klines_type=self.klines_type
        )

        self.datasets[pair_as_key] = {"sets": []}

        start = datetime.fromtimestamp(start_time)
        print(f"--- {round((time.time() - start_time), 1)}s seconds to retrieve Binance data on {pair_str} with interval {interval} request made at {start.strftime('%Y-%m-%d %H:%M:%S')} ---")

        return klines[1:] #.pop(0) # removing the first element (half-backed/ongoing current candle)


    ############################################
    # Generating Dataset Logic                 #
    ############################################

    def generate_concatinated_columns_for_dataset(self, cleaned_data):
        # Model features
        dataset = cleaned_data.copy()

        # Setup past related features by copying initial data and dropping unnecessary columns
        df_with_dropped_cols = cleaned_data.copy()
        del df_with_dropped_cols["close_time_dt_0"]
        del df_with_dropped_cols["was_up_0"]
        del df_with_dropped_cols["diff_0"]

        for i in range(1, self.candle_lookback_length):
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
        dataset = dataset.iloc[:(-1 * self.candle_lookback_length
)]

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

        # Can probably remove interval columns according to current file saving/naming conventions
        # for interval in self.possible_intervals:
        #     interval_series_column = dataset["close_time_dt_0"].apply(lambda x: 1 if interval == interval_of_interest else 0)
        #     interval_series_column.name = f"is_{interval}"

        #     interval_added_to_df = [dataset, interval_series_column]
        #     dataset = pd.concat(interval_added_to_df, axis=1)

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
            pair_sans_slash = pair.replace('/', '')
            pair_underscore = pair.replace('/', '_')

            for interval in self.intervals_of_interest:  
                klines = self.generate_klines(pair_sans_slash, interval, pair_as_key=pair_underscore)
                cleaned_data = self.clean_data(klines)
        
                dataset = self.generate_concatinated_columns_for_dataset(cleaned_data)    
                # dataset = self.generate_pairs_dataset(dataset, curr_pair_of_interest=pair_underscore)
                dataset = self.generate_time_info_for_dataset(dataset, interval)

                sets.append({
                    "interval": interval,
                    "dataset": dataset
                })


                temp_dataset = [self.full_dataset, dataset]
                self.full_dataset = pd.concat(temp_dataset, axis=0)

            self.datasets[pair_underscore]["sets"] = sets
        
        return [self.full_dataset, self.datasets]

    def __str__(self):
        return f"name: {self.name} klines_type: {self.klines_type}"