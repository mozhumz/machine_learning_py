from datetime import datetime
from datetime import date
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def job():
    print(1)

scheduler = BlockingScheduler()
scheduler.add_job(job, CronTrigger. from_crontab('*/1 * * * *'), id='my_job_id')
scheduler.start()
