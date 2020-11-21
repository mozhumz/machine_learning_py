import numpy as np
import math
arr1=np.array([1,2])
print(np.std(arr1))

# print(np.average(arr1))
#
# print((48+43+62+44)/4.)

def compute_std(arr:[]):
    sum_arr=0.
    for a in arr:
        sum_arr+=a
    avg=sum_arr/len(arr)
    sum_2=0.
    for a in arr:
        sum_2+=(a-avg)**2
    return math.sqrt(sum_2/len(arr))

print(compute_std(arr1))