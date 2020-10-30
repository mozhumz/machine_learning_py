import numpy as np
b = 9
a = 0.0 if b == 1 else 1

print(a)
ch_num=10
STATUS_NUM=4
arr=np.array([[0. for col in range(ch_num)] for row in range(STATUS_NUM)])
print(arr)

a2=[1,2]
b2={1,2}
print(type(a2))
print(type(b2))
print(1 in a2)
print(1 in b2)