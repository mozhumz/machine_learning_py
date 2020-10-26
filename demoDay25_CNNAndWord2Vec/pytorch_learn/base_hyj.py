# 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
# python2.X中print不需要括号，而在python3.X中则需要
from __future__ import print_function
import torch
import numpy as np
# 一个未初始化的5*3的矩阵
# x=torch.Tensor(5,3)
# print(x)

# 构建一个随机初始化的矩阵
x=torch.rand(5,3)
# print(x)
# print(x.size())

y=torch.rand(5,3)
# print(x+y)
# print(torch.add(x,y))

# res=torch.Tensor(5,3)
# torch.add(x,y,out=res)
# print(res)

y.add_(x)
# print(y)
# 取每行下标为1的列
# print(x[:,1])

'''
把一个torch张量转换为numpy数组或者反过来都是很简单的。
Torch张量和numpy数组将共享潜在的内存，改变其中一个也将改变另一个
'''
# a=torch.ones(5)
# print(a)
# b=a.numpy()
# print(b)
# print(type(b))

a=np.ones(5)
b=torch.from_numpy(a)
print(a,b)
# np.add(a,1,out=a)
b.add_(1)
print(a,b)

def cuda_test1(x,y):
    cuda_flag=torch.cuda.is_available()
    # cuda_flag=False
    if cuda_flag:
        x=x.cuda()
        y=y.cuda()
        print(x+y)

# cuda_test1(x,y)

'''
PyTorch 中所有神经网络的核心是autograd包.我们首先简单介绍一下这个包,然后训练我们的第一个神经网络.
autograd包为张量上的所有操作提供了自动求导.它是一个运行时定义的框架,这意味着反向传播是根据你的代码如何运行来定义,
并且每次迭代可以不同.
'''
from torch.autograd import Variable
x=Variable(torch.ones(2,2),requires_grad=True)
print(x)
y=x+2
print(y)