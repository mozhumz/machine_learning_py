import logging
import configparser
import time
import paramiko
import os
now_date=str(time.strftime('%Y-%m-%d'))
logging.basicConfig(level=logging.INFO,filename='logs/INFO'+now_date+'.log',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s : %(levelname)s : %(message)s')

config=configparser.ConfigParser()
config.read('conf.ini')
logging.info("start supervisor...")
try:
    ssh=paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect(hostname=config['worker']['hostname'],port=int(config['worker']['port']),
                username=config['worker']['username'],password=config['worker']['pwd'])
    # 注意可能要source java 或者 chmod 777 xxxdir
    cmd=('cd %s &&  java -jar %s &' %(config['app']['path'],config['app']['jar']))
    ssh.exec_command(cmd)
    # os.system(cmd)
except Exception as e:
    logging.warning('ssh-err: %s',e)


logging.info("supervisor done")