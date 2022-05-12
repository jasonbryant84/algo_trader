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

    def generate_sub_interval_data(self, interval_num, interval_unit):
        sub_interval = None
        sub_interval_unit = interval_unit
        offset = interval_num

        if interval_unit == "m":
            if interval_num == 1:
                sub_interval = 1
            elif interval_num == 5:
                sub_interval = 1
            elif interval_num == 15:
                sub_interval = 3
            elif interval_num == 30:
                sub_interval = 5

            offset = interval_num / sub_interval if interval_num != 1 else 0

        elif interval_unit == "h":
            if interval_num == 1:
                sub_interval = 15
                sub_interval_unit = "m"
                offset = 4
            elif interval_num == 4 or interval_num == 6:
                sub_interval = 1
                offset = interval_num / sub_interval
            elif interval_num == 12:
                sub_interval = 4
                offset = interval_num / sub_interval

        elif interval_unit == "d": # 1 day only implementation
            sub_interval = 4
            sub_interval_unit = "h"
            offset = 6
            
        elif interval_unit == "w":
            sub_interval = 1
            sub_interval_unit = "d"
            offset = 7

        elif interval_unit == "m":
            sub_interval = 1
            sub_interval_unit = "w"
            offset = 4

        else: # year
            sub_interval = 1
            sub_interval_unit = "m"
            offset = 12

        # start_time = f"{self.candle_lookback_length * interval_num} {unit} ago UTC"

        return sub_interval, sub_interval_unit, int(offset)


    def convertIntervals(self, interval):
        interval_unit = interval[-1]
        interval_num = int(interval[:-1])
        indicators_cushion = self.candle_lookback_length * 500 # cushion is in "interval_units"
        binance_cushion = 1 # api goes by datetime so adding one extra day/week/month cushion
        
        sub_interval, sub_interval_unit, offset = self.generate_sub_interval_data(interval_num, interval_unit)

        if interval_unit == "m":
            minutes_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion
            hours_total = int(math.ceil(minutes_total / 60)) + 1 
            days_total = int(math.ceil(minutes_total / 1440)) + binance_cushion
            return interval, f"{interval_num} minutes", f"{days_total} days ago UTC", sub_interval, sub_interval_unit, offset
        
        elif interval_unit == "h":
            hours_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion
            days_total = int(math.ceil(hours_total / 24)) + binance_cushion
            return interval, f"{interval_num} hours", f"{days_total} days ago UTC", sub_interval, sub_interval_unit, offset

        elif interval_unit == "d":
            days_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return interval, f"{interval_num} days", f"{days_total} days ago UTC", sub_interval, sub_interval_unit, offset
        
        elif interval_unit == "w":
            weeks_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return interval, f"{interval_num} weeks", f"{weeks_total} weeks ago UTC", sub_interval, sub_interval_unit, offset
        
        elif interval_unit == "M":
            months_total = ((self.candle_lookback_length + 1) * interval_num) + indicators_cushion + binance_cushion
            return interval, f"{interval_num} months", f"{months_total} months ago UTC", sub_interval, sub_interval_unit, offset

    def generate_indicators_for_dataset(self, df, prefix=""):
        df.insert(0, f"{prefix}ADX_20_0", ta.ADX(df[f"{prefix}high_0"], df[f"{prefix}low_0"], df[f"{prefix}close_0"], timeperiod=20))
        df.insert(0, f"{prefix}SMA_5_0", ta.SMA(df[f"{prefix}close_0"], 5))
        df.insert(0, f"{prefix}SMA_20_0", ta.SMA(df[f"{prefix}close_0"], 20))
        df.insert(0, f"{prefix}EMA_20_0", ta.EMA(df[f"{prefix}close_0"], 20))
        df.insert(0, f"{prefix}rsi_5_0", ta.RSI(df[f"{prefix}close_0"], 5))
        df.insert(0, f"{prefix}rsi_14_0", ta.RSI(df[f"{prefix}close_0"], 14))

        dif, dea, bar = ta.MACD(df[f"{prefix}close_0"].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df.insert(0, f"{prefix}macd_dif_0", dif)
        df.insert(0, f"{prefix}macd_dea_0", dea)
        df.insert(0, f"{prefix}macd_bar_0", bar)

        up, mid, low = ta.BBANDS(df[f"{prefix}close_0"], timeperiod = 20)
        df.insert(0, f"{prefix}low_band_20_0", low)
        df.insert(0, f"{prefix}mid_band_20_0", mid)
        df.insert(0, f"{prefix}up_band_20_0", up)

        # Visualize
        # df[['close_0', 'SMA_5_0', 'EMA_20_0', 'up_band_20_0', 'mid_band_20_0', 'low_band_20_0']].head(100).plot(figsize=(24,12))
        # plt.show()

        # df[['macd_dif_0', 'macd_dea_0', 'macd_bar_0']].head(100).plot(figsize=(24,12))
        # plt.show()
        
        return df.dropna()

    def clean_data(self, klines, sub_interval_offset):
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
        
        # Generating Indicators
        df = self.generate_indicators_for_dataset(df)

        # Calculate Open/Close Difference
        df.insert(0, "diff_0", df["close_0"] - df["open_0"])

        # Up/Down Label (no applid to sub interval dataset)
        df.insert(0, "was_up_0", df["diff_0"] > 0)
        df = df.astype({ "was_up_0": np.int8 })

        # Reverse row order (to create descending order)
        df = df.iloc[::-1]

        # Must be after generate_indicators_for_dataset due to NaN in "shift" column below
        if sub_interval_offset != 0:
            close_col = df["close_0"][:(-1*sub_interval_offset)]
            open_offset_col = df["close_0"].shift(-1*sub_interval_offset)[:(-1*sub_interval_offset)]

            df.insert(0, f"diff_shift{sub_interval_offset}_0", close_col - open_offset_col)

            # Drop last offset (number) rows
            df = df[:(-1*sub_interval_offset)]
    
            # Keep only ever offset-th row
            df = df.iloc[::sub_interval_offset, :]

            df.insert(0, f"was_up_shift{sub_interval_offset}_0", df[f"diff_shift{sub_interval_offset}_0"] > 0)
            df = df.astype({ f"was_up_shift{sub_interval_offset}_0": np.int8 })

        # Human readable datetime
        df.insert(0, "close_time_dt_0", pd.to_datetime(df["close_time_0"], unit='ms'))

        # Drop Open and Close times and Ignore column (maybe this is important info from Binance)
        del df["open_time_0"]
        del df["close_time_0"]
        del df["ignore_0"]

        print(f"--- {round((time.time() - start_time), 1)}s seconds to clean data ---")
        return df


    def generate_klines(self, pair_str, interval, pair_as_key, use_sub_intervals=False):
        start_time = time.time()

        _, _, start_str, sub_interval, sub_interval_unit, offset  = self.convertIntervals(interval)
        
        interval = f"{sub_interval}{sub_interval_unit}" if use_sub_intervals else interval
        offset = offset if use_sub_intervals else 0

        print(f"get_historical_klines starting: {datetime.utcnow()}")
        klines = self.client.get_historical_klines(
            symbol=pair_str,
            interval=interval,
            start_str=start_str,
            klines_type=self.klines_type
        )
        print(f"get_historical_klines ending:   {datetime.utcnow()}")

        self.datasets[pair_as_key] = {"sets": []}

        start = datetime.fromtimestamp(start_time)
        print(f"--- {round((time.time() - start_time), 1)}s seconds to retrieve Binance data on {pair_str} with interval {interval} request made at {start.strftime('%Y-%m-%d %H:%M:%S')} ---")

        
        return klines[:-1], offset # drop last kline as it's half-baked (it's the most recent candle, descending order klines)


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

            # Drop first row (shift every row up by 1)
            df_with_dropped_cols = df_with_dropped_cols.shift(periods=-1, axis="rows")

            # Concatenate
            updated_df = [dataset, df_with_dropped_cols]
            dataset = pd.concat(updated_df, axis=1)

        # Dropping the last "candle_lookback_length" rows as they will have NaN values due to shifting
        dataset = dataset.iloc[:(-1 * self.candle_lookback_length)]

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

    def generate_datasets(self, use_sub_intervals=False):    
        for pair in self.pairs_of_interest:
            sets = [] 
            pair_sans_slash = pair.replace('/', '')
            pair_underscore = pair.replace('/', '_')

            for interval in self.intervals_of_interest:  
                klines, offset = self.generate_klines(
                    pair_sans_slash,
                    interval,
                    pair_as_key=pair_underscore,
                    use_sub_intervals=use_sub_intervals
                )

                cleaned_data = self.clean_data(
                    klines,
                    sub_interval_offset=offset
                )
        
                dataset = self.generate_concatinated_columns_for_dataset(cleaned_data)  
                # dataset = self.generate_pairs_dataset(dataset, curr_pair_of_interest=pair_underscore)
                dataset = self.generate_time_info_for_dataset(dataset, interval)

                dataset = dataset.reset_index(drop=True)

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