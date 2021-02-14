'''
在这个教程中,我们使用CIFAR10数据集,它有如下10个类别:
'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'.
这个数据集中的图像大小为3*32*32,即,3通道,32*32像素
'''

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
torchvision的输出是[0,1]的PILImage图像,我们把它转换为归一化范围为[-1, 1]的张量
'''
# pre='F:\\八斗学院\\视频\\14期正式课\\00-data\\CIFAR10'
pre = 'G:\\bigdata\\badou\\00-data\\CIFAR10'
# 加载数据
tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=pre, train=True, download=True, transform=tranform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=pre, train=True, download=True, transform=tranform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 展示图片
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 展示一些训练图像.
dataiter = iter(trainloader)
imgs, labels = dataiter.next()
imshow(torchvision.utils.make_grid(imgs))
print(' '.join('%s' % classes[labels[j]] for j in range(4)))


# 定义一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
Loss_fn = nn.CrossEntropyLoss()
eta = 0.001
optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)
#  训练网络
# gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
for epoch in range(2):
    loss_val = 0.
    for i, data in enumerate(trainloader, start=0):
        inputs, labels = data
        # 样本转换为Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # gpu训练
        # inputs, labels = data[0].to(device), data[1].to(device)
        # 参数梯度重置为0
        optimizer.zero_grad()
        # 每次迭代的输出
        outputs = net(inputs)
        # 每次迭代的损失
        loss = Loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val += loss.data[0]
        # print every 2000 mini-batches
        if i%2000==1999:
            # %5d 表示参数间有5个空格 %.3f表示四舍五入保留3为小数
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss_val / 2000))
            loss_val = 0.0

print('Finished Training')

# 在测试集上测试网络
dataiter = iter(testloader)
# 测试集中某些图片
test_images, test_labels = dataiter.next()

# 测试集中某些图片的真实分类 print images
imshow(torchvision.utils.make_grid(test_images))
print('GroundTruth: ', ' '.join('%5s' % classes[test_labels[j]] for j in range(4)))

# 测试集中某些图片的预测分类
pred_out=net(Variable(test_images))
_, predicted = torch.max(pred_out.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 整个测试集的auc
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# 那在什么类上预测较好,什么类预测结果不好呢
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))