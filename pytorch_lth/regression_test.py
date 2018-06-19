# coding=utf-8
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F # 函数部分
import torchvision
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = torch.autograd.Variable(x), Variable(y) #将x,y转换为pytorch能够处理的Variable类型

# 网络结构模块
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__() #继承__init__功能
        self.hidden=torch.nn.Linear(n_feature,n_hidden) #全连接层
        self.predict=torch.nn.Linear(n_hidden,n_output) #输出层

    def forward(self,x):
        #正向传播输入值
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=10,n_output=1)

optimizer=torch.optim.SGD(net.parameters(),lr=0.1) #传入网络参数与学习率
loss_f=torch.nn.MSELoss() #损失函数

plt.ion()   # 画图
plt.show()

for t in range(5000):
    prediction=net(x)
    loss=loss_f(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.text(0,0.5,'times=%d' %t,fontdict={'size': 20, 'color': 'blue'})
        plt.pause(0.1)
print(net)
