from concurrent.futures import ThreadPoolExecutor
threadPool = ThreadPoolExecutor(max_workers=100)


def test1():
    relist=[1,2,3,4,5]
    print(relist[:3])
    print(relist[3:])

if __name__ == '__main__':
    threadPool.submit(test1())
    print('ok')

