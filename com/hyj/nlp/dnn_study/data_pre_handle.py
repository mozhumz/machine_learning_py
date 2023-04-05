#%%

import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

#%%

# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)

#%%

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

#%%

inputs,outputs

#%%

c=inputs.iloc[:,0:1]
c

#%%

c=c.fillna(c.mean())
c

#%%

inputs.iloc[:,0:1]=c

#%%

inputs

#%%

c=inputs.iloc[:,0]

#%%

c

#%%

c.mean()
inputs.iloc[:,0]=c.fillna(c.mean())

#%%

inputs.iloc[:,0]

#%%

inputs

#%%

inputs=pd.get_dummies(inputs,dummy_na=True)*1
inputs

#%%



#%%

import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y

#%%


