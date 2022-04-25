import datetime, subprocess
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sched = BlockingScheduler()
sched.configure(timezone=utc)

# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute='4-59/5', second='56')
def predict_5min_interval():
    now = datetime.datetime.utcnow()
    print (f"timed_job_5min_interval(): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # trade.py --pair XRP/USDT --interval 5m --candles 50 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', 'XRP/USDT', '--interval', '5m', '--candles', '50', '--loadCloudModel'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute='*/30')
def generate_model_30min_after_the_hour():
    now = datetime.datetime.utcnow()
    print (f"generate_model_30min_after_the_hour(): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python algo_trader.py --pair XRP/USDT --interval 5m --candles 50 --epochs 1 --learning_rate 0.03  --cloudStorage
    p = subprocess.run(['python', 'algo_trader.py', '--pair', 'XRP/USDT', '--interval', '5m', '--candles', '50', '--epochs', '18', '--learning_rate 0.03', '--cloudStorage'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute='*/15')
def update_predictions_metrics():
    now = datetime.datetime.utcnow()
    print (f"update_predictions_metrics(): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python check_trades.py --pair XRP/USDT --interval 5m --candles 50 --cloudStorage
    p = subprocess.run(['python', 'check_trades.py', '--pair', 'XRP/USDT', '--interval', '5m', '--candles', '50', '--cloudStorage'])


sched.start()