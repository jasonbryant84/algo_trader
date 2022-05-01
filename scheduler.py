import datetime, subprocess
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sched = BlockingScheduler()
sched.configure(timezone=utc)

pair = 'XRP/USDT'
interval_str = '5m'
interval_num = interval_str[:-1]
n_candles = '10'
n_epochs = '33'
learning_rate = '0.01'

# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute=f"{int(interval_num) - 1}-59/{interval_num}", second='56')
def predict_on_interval():
    now = datetime.datetime.utcnow()
    print (f"--- predict_on_interval ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python trade.py --pair XRP/USDT --interval 5m --candles 10 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--loadCloudModel'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute=f"*/{int(interval_num)*3}", second='59')
def generate_model_30min_after_the_hour():
    now = datetime.datetime.utcnow()
    print (f"generate_model_30min_after_the_hour ({interval_str} interva): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python setup.py --pair XRP/USDT --interval 5m --candles 10 --epochs 33 --learning_rate 0.03  --noStorage
    p = subprocess.run(['python', 'setup.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--epochs', n_epochs, '--learning_rate', learning_rate, '--noStorage'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute=f"57") # at 57 of every hour
def update_predictions_metrics():
    now = datetime.datetime.utcnow()
    print (f"--- update_predictions_metrics ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python check_trades.py --pair XRP/USDT --interval 5m --candles 50 --cloudStorage
    p = subprocess.run(['python', 'check_trades.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--cloudStorage'])


sched.start()