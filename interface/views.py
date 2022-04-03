import json
from django.conf import settings
from django.http import HttpResponse
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from .utils.helpers import BinanceHelper

client = Client(settings.BINANCE_API_KEY, settings.BINANCE_SECRET)

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def binance(request):
    pair = request.GET.get('pair')
    parin_sans_slash = pair.replace('/','')
    interval = request.GET.get('interval')
    limit = request.GET.get('limit')

    # get market depth
    depth = client.get_order_book(symbol=parin_sans_slash)

    # place a test market buy order, to place an actual order use the create_order function
    # order = client.create_test_order(
    #     symbol='BNBBTC',
    #     side=Client.SIDE_BUY,
    #     type=Client.ORDER_TYPE_MARKET,
    #     quantity=100)

    # get all symbol prices
    prices = client.get_all_tickers()
    pair_price = [item for item in prices if item.get('symbol')==parin_sans_slash]
    print('pair_price', pair_price[0])

    # # withdraw 100 ETH
    # # check docs for assumptions around withdrawals
    # from binance.exceptions import BinanceAPIException
    # try:
    #     result = client.withdraw(
    #         asset='ETH',
    #         address='<eth_address>',
    #         amount=100)
    # except BinanceAPIException as e:
    #     print(e)
    # else:
    #     print("Success")

    # # fetch list of withdrawals
    # withdraws = client.get_withdraw_history()

    # # fetch list of ETH withdrawals
    # eth_withdraws = client.get_withdraw_history(coin='ETH')

    # # get a deposit address for BTC
    # address = client.get_deposit_address(coin='BTC')

    # # get historical kline data from any date range

    # # fetch klines
    # list of OHLCV values (Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore)
    helper = BinanceHelper(pair, interval, limit)
    print('helper', helper)
    helper.generate_klines()
    
    # print('klines', klines)

    # # fetch 30 minute klines for the last month of 2017
    # klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

    # # fetch weekly klines since it listed
    # klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

    # # socket manager using threads
    # twm = ThreadedWebsocketManager()
    # twm.start()

    # # depth cache manager using threads
    # dcm = ThreadedDepthCacheManager()
    # dcm.start()

    # def handle_socket_message(msg):
    #     print(f"message type: {msg['e']}")
    #     print(msg)

    # def handle_dcm_message(depth_cache):
    #     print(f"symbol {depth_cache.symbol}")
    #     print("top 5 bids")
    #     print(depth_cache.get_bids()[:5])
    #     print("top 5 asks")
    #     print(depth_cache.get_asks()[:5])
    #     print("last update time {}".format(depth_cache.update_time))

    # twm.start_kline_socket(callback=handle_socket_message, symbol='BNBBTC')

    # dcm.start_depth_cache(callback=handle_dcm_message, symbol='ETHBTC')

    # # replace with a current options symbol
    # options_symbol = 'BTC-210430-36000-C'
    # dcm.start_options_depth_cache(callback=handle_dcm_message, symbol=options_symbol)

    # # join the threaded managers to the main thread
    # twm.join()
    # dcm.join()

    return HttpResponse(f"{pair_price[0]} {len(helper.klines)} {helper.klines[0]}")