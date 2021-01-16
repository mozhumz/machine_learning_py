'''
线程池测试
'''
from concurrent.futures import ThreadPoolExecutor
import time
import threading

lockVal= []
global lockVal
lock=threading.RLock()

def test(value1='v1', value2='v2'):
    print("%s threading is printed %s, %s"%(threading.current_thread().name, value1, value2))
    time.sleep(2)
    lockAndAdd()
    return 'finished'

def test_result(future):
    print(future.result())

def lockAndAdd():
    lock.acquire(timeout=5000)
    try:
        lockVal.append(1)
    finally:
        lock.release()

if __name__ == "__main__":

    threadPool = ThreadPoolExecutor(max_workers=4)
    for i in range(0,10):
        future = threadPool.submit(test)

    threadPool.shutdown(wait=True)
    print(lockVal)