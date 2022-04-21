import datetime, subprocess
from pytz import utc

import trade

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sched = BlockingScheduler()

executors = {
    'default': ThreadPoolExecutor(20),
    'processpool': ProcessPoolExecutor(5)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 3
}
sched.configure(
    executors=executors,
    job_defaults=job_defaults,
    timezone=utc
)

# 2 seconds before every 5 minute interval
@sched.scheduled_job('cron', minute='4-59/5', second='58')
def timed_job():
    now = datetime.datetime.utcnow()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    # trade.py --pair XRP/USDT --interval 5m --candles 50 --loadCloudModel  
    p = subprocess.run(['python', 'trade.py', '--pair', 'XRP/USDT', '--interval', '5m', '--candles', '50', '--loadCloudModel'])

sched.start()