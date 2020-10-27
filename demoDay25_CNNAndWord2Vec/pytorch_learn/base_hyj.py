# 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
# python2.X中print不需要括号，而在python3.X中则需要
from __future__ import print_function
import torch
import numpy as np

# 一个未初始化的5*3的矩阵
# x=torch.Tensor(5,3)
# print(x)

# 构建一个随机初始化的矩阵
x = torch.rand(5, 3)
# print(x)
# print(x.size())

y = torch.rand(5, 3)
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

a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
# np.add(a,1,out=a)
b.add_(1)
print(a, b)


def cuda_test1(x, y):
    cuda_flag = torch.cuda.is_available()
    # cuda_flag=False
    if cuda_flag:
        x = x.cuda()
        y = y.cuda()
        print(x + y)


# cuda_test1(x,y)

'''
PyTorch 中所有神经网络的核心是autograd包.我们首先简单介绍一下这个包,然后训练我们的第一个神经网络.
autograd包为张量上的所有操作提供了自动求导.它是一个运行时定义的框架,这意味着反向传播是根据你的代码如何运行来定义,
并且每次迭代可以不同.
'''
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
# print(x)
y = x + 2
# print(y)

z = y * y * 3
out = z.mean()
# print(z,out)

# 自动求导
out.backward()
# print(x.grad)

x = torch.ones([2,2],requires_grad=True)
y = x * 3
# y.data.norm()其实就是对y张量L2范数，先对y中每一项取平方，之后累加，最后取根号
# print(torch.sqrt(torch.sum(torch.pow(y,2))))
# while y.data.norm()<1000:
#     y*=2
    # print('while:',y)
print(y)

v=torch.tensor([[0.1,1],[1,1]])
# v表示梯度张量 x.grad表示y对x求导后乘以梯度v
y.backward(v)
print(x.grad)

m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True)
# n = Variable(torch.zeros(1, 2))
# n[0, 0] = m[0, 0] ** 2
# n[0, 1] = m[0, 1] ** 3
# print(m.size(),n.size())
# n.backward(torch.tensor([[1,1]]))
# print(m.grad)

j = torch.zeros(2 ,2)
k = Variable(torch.zeros(1, 2))
# 在grad更新时，每一次运算后都需要将上一次的梯度记录清空
# m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
k.backward(torch.FloatTensor([[1, 0]]),retain_graph=True)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(torch.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('jacobian matrix is')
# 雅克比矩阵
print(j)

print(x.requires_grad)
print((x ** 2).requires_grad)

# with torch.no_grad():
#     print((x ** 2).requires_grad)

y=x.detach()
print(y.requires_grad)
print(x.eq(y).all())

target=torch.randn(1,10)
print(target)
target=target.view(1,-1)
print(target)
print(v)
print(v.add_(v*0.1))

print('[%d, %d] loss: %.3f' %
      (1 + 1, 3 + 1, 5.68953))