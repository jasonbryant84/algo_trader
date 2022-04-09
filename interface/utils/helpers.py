import time

from datetime import datetime
from django.conf import settings

import pandas as pd
import numpy as np

from binance import Client
from binance.enums import HistoricalKlinesType
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from .taapi import TaapiInterface

class CryptoPair:
    def __init__(self):
        self.name = 'Crypto Name'
        self.df = None

class ExchangeHelper:
    def __init__(self):
        self.name = 'Exchange Helper'
        self.client = None

    class Meta:
        abstract = True

class BinanceHelper(ExchangeHelper):
    name = 'Binance Helper'
    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_SECRET)
    
    def __init__(self, pair, interval, klines_type='spot'):
        self.pair = pair
        self.pair_sans_slash = pair.replace('/', '')
        self.interval = interval
        self.klines_intervals = self.convertIntervals()[0]
        self.klines_start_str = self.convertIntervals()[2]
        self.klines_type = HistoricalKlinesType.SPOT if klines_type != 'futures' else HistoricalKlinesType.FUTURES

        self.klines = []
        self.rsi = []
        self.macd = []
        self.ema5 = []
        self.ema20 = []
        self.bbands = []

        self.crypto_pair = None
        self.features = None

    def convertIntervals(self):
        if self.interval == '1m':
            return [Client.KLINE_INTERVAL_1MINUTE, '1 minute', '2 days ago UTC']
        elif self.interval == '15m':
            return [Client.KLINE_INTERVAL_15MINUTE, '15 minutes', '2 months ago UTC']
        elif self.interval == '1h':
            return [Client.KLINE_INTERVAL_1HOUR, '1 hour', '6 months ago UTC']
        elif self.interval == '4h':
            return [Client.KLINE_INTERVAL_4HOUR, '4 hours', '2 years ago UTC']
        elif self.interval == '1d':
            return [Client.KLINE_INTERVAL_1DAY, '1 day', '5 years ago UTC']
        elif self.interval == '1w':
            return [Client.KLINE_INTERVAL_1WEEK, '1 week', '9 years ago UTC']
        elif self.interval == '1M':
            return [Client.KLINE_INTERVAL_1MONTH, '1 month', '12 years ago UTC']
        else:
            return [Client.KLINE_INTERVAL_1DAY, '1 day (default)', '1 day ago UTC']

    def clean_data(self):
        start_time = time.time()
        self.crypto_pair = CryptoPair()
        self.crypto_pair.df = pd.DataFrame(
            self.klines,
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
        self.crypto_pair.df = self.crypto_pair.df.astype(new_dtypes)

        # Calculate Open/Close Difference
        self.crypto_pair.df.insert(0, "diff_0", self.crypto_pair.df["close_0"] - self.crypto_pair.df["open_0"])

        # Up/Down
        self.crypto_pair.df.insert(0, "was_up_0", self.crypto_pair.df["diff_0"] > 0)
        self.crypto_pair.df = self.crypto_pair.df.astype({ "was_up_0": int })

        # Human readable datetime
        self.crypto_pair.df.insert(0, "close_time_dt_0", pd.to_datetime(self.crypto_pair.df["close_time_0"], unit='ms'))

        # Drop Open and Close times and Ignore column (maybe this is important info from Binance)
        del self.crypto_pair.df["open_time_0"]
        del self.crypto_pair.df["close_time_0"]
        del self.crypto_pair.df["ignore_0"]

        # Reverse row order
        self.crypto_pair.df = self.crypto_pair.df.iloc[::-1]

        # taapi_interface = TaapiInterface()
        # request = taapi_interface.rsi(self.pair, self.interval)

        print("--- %s seconds to clean data ---" % (time.time() - start_time))
        return self.crypto_pair.df


    def generate_klines(self):
        start_time = time.time()
        self.klines = self.client.get_historical_klines(
            symbol=self.pair_sans_slash,
            interval=self.klines_intervals,
            start_str=self.klines_start_str,
            end_str=None,
            klines_type=self.klines_type
        )

        print("--- %s seconds to retrieve Binance data ---" % (time.time() - start_time))
        return self.klines


    def generate_features(self):
        lookback_length = 100

        # Model features
        self.features = self.crypto_pair.df.copy()

        # Setup past related features by copying initial data and dropping unnecessary columns
        df_with_dropped_cols = self.crypto_pair.df.copy()
        del df_with_dropped_cols["close_time_dt_0"]
        del df_with_dropped_cols["was_up_0"]
        del df_with_dropped_cols["diff_0"]

        for i in range(1, lookback_length):
            prevIStr = f"_{i-1}"
            prevSuffixOffset = -1 * len(prevIStr)

            # Update suffix to indicate past index
            df_with_dropped_cols = df_with_dropped_cols.rename(columns = lambda x : str(x)[:prevSuffixOffset]).add_suffix(f"_{i}")

            # Drop first row
            df_with_dropped_cols = df_with_dropped_cols.shift(periods=-1, axis="rows")

            # Concatenate
            updated_df = [self.features, df_with_dropped_cols]
            self.features = pd.concat(updated_df, axis=1)

        # Dropping the last "lookback_length" rows as they will have NaN values due to shifting
        self.features = self.features.iloc[:(-1 * lookback_length)]
        print('self.features\n', self.features)

    def __str__(self):
        return f"name: {self.name} kline_intervals: {self.klines_intervals} klines_type: {self.klines_type}"