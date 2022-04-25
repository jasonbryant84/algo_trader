import datetime, subprocess
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sched = BlockingScheduler()
sched.configure(timezone=utc)

pair = 'XRP/USDT'
interval_str = '5m'
interval = int(interval_str[:-1])
n_candles = '50'
n_epochs = 18
learning_rate = 0.03

# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute=f"4-59/{interval}", second='56')
def predict_on_interval():
    now = datetime.datetime.utcnow()
    print (f"--- predict_on_interval ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # trade.py --pair XRP/USDT --interval 5m --candles 50 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--loadCloudModel'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute='*/30')
def generate_model_30min_after_the_hour():
    now = datetime.datetime.utcnow()
    print (f"generate_model_30min_after_the_hour ({interval_str} interva): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python algo_trader.py --pair XRP/USDT --interval 5m --candles 50 --epochs 1 --learning_rate 0.03  --cloudStorage
    p = subprocess.run(['python', 'algo_trader.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--epochs', n_epochs, '--learning_rate', learning_rate, '--cloudStorage'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute=f"*/{interval}", second='30')
def update_predictions_metrics():
    now = datetime.datetime.utcnow()
    print (f"--- update_predictions_metrics ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python check_trades.py --pair XRP/USDT --interval 5m --candles 50 --cloudStorage
    p = subprocess.run(['python', 'check_trades.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--cloudStorage'])


sched.start()