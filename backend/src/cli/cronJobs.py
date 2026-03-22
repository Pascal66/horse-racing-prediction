import os
import sys
from datetime import datetime, timedelta
import schedule

from backend.src.cli.etl import etl_daily, etl_liveodds

# tasks.scan.triggers=daily:7h30
# tasks.scan.dayoffsets=-1,0,1
# tasks.odds.triggers=-6h,-15m,-3m,0m,+1m*
# tasks.better.triggers=-5m
# tasks.finish.triggers=+30m,+1h*
def cronjobs():

    TODAY = datetime.today().strftime("%d%m%Y")
    START_DATE = (datetime.today() - timedelta(days=2)).strftime("%d%m%Y")
    END_DATE = (datetime.today() + timedelta(days=1)).strftime("%d%m%Y")

    job1 = schedule.every().day.at('07:30').do(etl_daily, START_DATE, END_DATE).tag('daily-tasks', 'Orchestrator')
    job2 = schedule.every(15).minutes.do(etl_liveodds).tag('odds-tasks', 'Orchestrator')

if __name__ == "__main__":
    pass
    # TODAY = datetime.today().strftime("%d%m%Y")
    # START_DATE = (datetime.today() - timedelta(days=2)).strftime("%d%m%Y")
    # END_DATE = (datetime.today() + timedelta(days=1)).strftime("%d%m%Y")
    #
    # job = schedule.every().day.at('07:30').do(etl_daily, START_DATE, END_DATE).tag('daily-tasks', 'Orchestrator')
    #
    # print(f'job {schedule.get_jobs()} {START_DATE} to {END_DATE}')
    # schedule.run_all()