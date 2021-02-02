'''
线程池测试
'''
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import logging

lockVal= []
global lockVal
lock=threading.RLock()

def test(value1='v1', value2='v2',log=None):
    print("%s threading is printed %s, %s"%(threading.current_thread().name, value1, value2))
    time.sleep(2)
    lockAndAdd(log)
    return 'finished'

def test_result(future):
    print(future.result())

def lockAndAdd(log=None):
    lock.acquire(timeout=5000)
    try:
        lockVal.append(1)
        log.info('lockAndAdd:'+threading.current_thread().name)
    finally:
        lock.release()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('start')
    threadPool = ThreadPoolExecutor(max_workers=4)
    for i in range(0,10):
        future = threadPool.submit(test,log=logger)

    threadPool.shutdown(wait=True)
    print(lockVal)