import datetime as dt
import logging
import time
import os


def date_hash():
    return dt.datetime.now().strftime('%y%m%d%H%M') + str(hash(time.time()))[:4]


def print_log(fname, content):
    logging.basicConfig(filename=os.path.join(fname + '.log'), level=logging.ERROR)
    logger = logging.getLogger('model')
    logger.setLevel(logging.INFO)
    print(content)
    logger.info(content)