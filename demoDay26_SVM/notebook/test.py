from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
import math
# data = load_breast_cancer()
# X = data.data
# print(X.shape) # (569,30)
# y = data.target
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

gamma_range = np.logspace(2,3,3)
print("gamma_range: ",gamma_range)
print(math.pow(10,2.5))