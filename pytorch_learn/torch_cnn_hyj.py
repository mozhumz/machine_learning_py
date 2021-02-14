import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_hyj(nn.Module):
    def __init__(self):
        super(Net_hyj, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 这个是把一个矩阵重新排列成不同维度但不改变元素的函数，
        # 如代码中x.view(-1,self.num_flat_features(x))，这里-1就是把后面的矩阵展成一维数组，以便后面线性变换层操作
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_feat = 1
        for s in size:
            num_feat *= s
        return num_feat


net = Net_hyj()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
'''
注意
torch.nn 只支持小批量输入,整个torch.nn包都只支持小批量样本,而不支持单个样本
例如,nn.Conv2d将接受一个4维的张量,每一维分别是sSamples * nChannels * Height * Width(样本数*通道数*高*宽).
如果你有单个样本,只需使用input.unsqueeze(0)来添加其它的维数.
'''
input = torch.randn(1, 1, 28, 28)
out = net(input)
print(out)
target = torch.randn(1, 10)
target = target.view(1, -1)
mse = nn.MSELoss()
# loss = mse(out, target)
# loss2 = torch.mean(torch.pow(out - target, 2))
# print(loss, loss2)
# # mse
# print(loss.grad_fn)
# # linear
# print(loss.grad_fn.next_functions[0][0])
# # ReLu
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
#
# net.zero_grad()
# print('conv1-b-bp_before:', net.conv1.bias.grad)
# loss.backward()
# print('conv1-b-bp_after:', net.conv1.bias.grad)
#
# eta = 0.01
# total_params = 10
# i = 0
# for f in net.parameters():
#     i += 1
#     if i == total_params:
#         print('ok')
#     print('before:', f.data)
#     # sub_ 减法
#     f.data.sub_(f.grad.data * eta)
#     print('after:', f.data)
#     print('i:',i)
def print_params(net):
    for f in net.parameters():
        print(f.data)

import torch.optim as optim
print('before--------------------------')
print_params(net)
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
loss=mse(out,target)
loss.backward()
optimizer.step()
print('after--------------------------------')
print_params(net)

