import datetime, subprocess
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sched = BlockingScheduler()
sched.configure(timezone=utc)

# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute='4-59/2', second='58')
def timed_job():
    now = datetime.datetime.utcnow()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    # trade.py --pair XRP/USDT --interval 5m --candles 50 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', 'XRP/USDT', '--interval', '5m', '--candles', '50', '--loadCloudModel'])

sched.start()