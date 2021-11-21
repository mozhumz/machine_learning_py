import logging
import configparser
import time
now_date=str(time.strftime('%Y-%m-%d'))
logging.basicConfig(level=logging.INFO,filename='logs/INFO'+now_date+'.log',datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)s - %(module)s - %(message)s')
logging.info('------------------------------start--------------------------')
conf=configparser.ConfigParser()
conf.read('config.ini')
path=conf['app']['path']
print(path)
logging.info('path %s',path)
logging.info('----------------------------end-------------------------')