import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2],[1, 1, 1]])
cost = np.array([[4, 1, 3,1], [2, 0, 5,2], [3, 2, 2,3]])
'''
col1 col2 col3 col4
1       1   1   1
2       2   2   2
3       3   3   3
'''
df=pd.DataFrame.from_dict({'col1':[1,2,1],'col2':[3,2,3],'col3':[4,3,2],'col4':[2,3,3]})
row_ind, col_ind = linear_sum_assignment(df)

print(row_ind,col_ind)