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
        self.features = None

class ExchangeHelper:
    def __init__(self):
        self.name = 'Exchange Helper'
        self.client = None

    class Meta:
        abstract = True

class BinanceHelper(ExchangeHelper):
    name = 'Binance Helper'
    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_SECRET)
    
    def __init__(self, pair, interval, limit, klines_type='spot'):
        self.pair = pair
        self.pair_sans_slash = pair.replace('/', '')
        self.interval = interval
        self.klines_intervals = self.convertIntervals()[0]
        self.klines_limit = int(limit)
        self.klines_type = HistoricalKlinesType.SPOT if klines_type != 'futures' else HistoricalKlinesType.FUTURES

        self.klines = []
        self.rsi = []
        self.macd = []
        self.ema5 = []
        self.ema20 = []
        self.bbands = []

    def convertIntervals(self):
        if self.interval == '1m':
            return [Client.KLINE_INTERVAL_1MINUTE, '1 minute']
        elif self.interval == '1h':
            return [Client.KLINE_INTERVAL_1HOUR, '1 hour']
        elif self.interval == '1d':
            return [Client.KLINE_INTERVAL_1DAY, '1 day']
        elif self.interval == '1w':
            return [Client.KLINE_INTERVAL_1WEEK, '1 week']
        elif self.interval == '1M':
            return [Client.KLINE_INTERVAL_1MONTH, '1 month']
        else:
            return [Client.KLINE_INTERVAL_1DAY, '1 day (default)']

    def clean_data(self):
        pair_data = CryptoPair()
        pair_data.features = pd.DataFrame(
            self.klines,
            columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset vol', 'Num trades', 'Taker buy base asset vol', 'Taker buy quote asset vol', 'Ignore'] # list of OHLCV
        )

        del pair_data.features["Ignore"]

        pair_data.features["Close time dt"] = pd.to_datetime(pair_data.features["Close time"], unit='ms')

        taapi_interface = TaapiInterface()
        request = taapi_interface.rsi(self.pair, self.interval)

        return pair_data.features


    def generate_klines(self):
        self.klines = self.client.get_historical_klines(
            symbol=self.pair_sans_slash,
            interval=self.klines_intervals,
            start_str="2018-01-01",
            end_str=None,
            limit=self.klines_limit,
            klines_type=self.klines_type
        )

        print('test\n', self.clean_data())
        return self.klines


    def __str__(self):
        return f"name: {self.name} kline_intervals: {self.klines_intervals} limit: {self.klines_limit} klines_type: {self.klines_type}"