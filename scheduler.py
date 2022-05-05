import datetime, subprocess, argparse
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

parser = argparse.ArgumentParser(description="Scheduled scripts")
parser.add_argument("--pair", dest="pair", default="XRP/USDT", help="traiding pair")
parser.add_argument("--interval", dest="interval", default="5m", help="time interval")
parser.add_argument("--candles", dest="n_candles", default="10", help="number of candles for look back")
parser.add_argument("--epochs", dest="n_epochs", default="35", help="number of epochs use to train the neural network")
parser.add_argument("--learning_rate", dest="learning_rate", default="0.01", help="learning rate to be used to train the neural network")
args = parser.parse_args()

sched = BlockingScheduler()
sched.configure(timezone=utc)

pair = args.pair or 'XRP/USDT'
interval_str = args.interval or '5m'
interval_num = interval_str[:-1]
n_candles = args.n_candles or '10'
n_epochs = args.n_epochs or '35'
learning_rate = args.learning_rate or '0.01'
# python scheduler.py --pair XRP/USDT --interval 5m --candles 10 --epochs 25 --learning_rate 0.01


# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute=f"{int(interval_num) - 1}-59/{interval_num}", second='59')
def predict_on_interval():
    now = datetime.datetime.utcnow()
    print (f"--- predict_on_interval ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python trade.py --pair XRP/USDT --interval 5m --candles 10 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--loadCloudModel'])


# Generate a model 30 minutes after every hour
@sched.scheduled_job('cron', minute=f"56")
def generate_model():
    now = datetime.datetime.utcnow()
    print (f"generate_model ({interval_str} interva): {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # python setup.py --pair XRP/USDT --interval 5m --candles 10 --epochs 25 --learning_rate 0.01  --noStorage --liveMode
    p = subprocess.run(['python', 'setup.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--epochs', n_epochs, '--learning_rate', learning_rate, '--noStorage', '--liveMode'])


# # Generate a model 30 minutes after every hour
# @sched.scheduled_job('cron', minute=f"57") # at 57 of every hour
# def update_predictions_metrics():
#     now = datetime.datetime.utcnow()
#     print (f"--- update_predictions_metrics ({interval_str} interval): {now.strftime('%Y-%m-%d %H:%M:%S')}")

#     # python check_trades.py --pair XRP/USDT --interval 5m --candles 10 --cloudStorage
#     p = subprocess.run(['python', 'check_trades.py', '--pair', pair, '--interval', interval_str, '--candles', n_candles, '--cloudStorage'])


sched.start()