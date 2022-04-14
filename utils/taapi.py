import requests

import pandas as pd
import numpy as np

class TaapiInterface:
    def __init__(self, exchange = 'binance'):
        self.api_key = settings.TAAPI_API_KEY
        self.exchange = exchange

    def get(self, url, params):
        return requests.get(url = url, params = params)

    def rsi(self, symbol, interval, backtracks=1):
        url = f"https://api.taapi.io/rsi?secret={self.api_key}&exchange={self.exchange}&symbol={symbol}&interval={interval}&backtracks={backtracks}"

        print('url', url)
        return self.get(url, {})