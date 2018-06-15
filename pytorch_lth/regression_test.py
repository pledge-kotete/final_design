# coding=utf-8
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# data=np.arange(200)
# # tensor=torch.FloatTensor(data)
# # torch.get_default_dtype()
# # print(torch.sin(tensor))
# x=torch.linspace(-5,5,200)
# weight=torch.FloatTensor(data)
# x=Variable(x)
#
# x_np=x.data.numpy()
#
# y_relu=F.relu(x).data.numpy()
# y_prelu=F.prelu(x,weight).data.numpy()
# y_sigmoid=F.sigmoid(x).data.numpy()
# y_tanh=F.tanh(x).data.numpy()
# y_softplus=F.softplus(x).data.numpy()
#
# plt.figure(1, figsize=(8, 6))
# plt.subplot(321)
# plt.plot(x_np, y_relu, c='red', label='relu')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
#
# plt.subplot(322)
# plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
# plt.ylim((-0.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(323)
# plt.plot(x_np, y_tanh, c='red', label='tanh')
# plt.ylim((-1.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(324)
# plt.plot(x_np, y_softplus, c='red', label='softplus')
# plt.ylim((-0.2, 6))
# plt.legend(loc='best')
#
# plt.subplot(325)
# plt.plot(x_np, y_prelu, c='red', label='prelu')
# # plt.ylim((-0.2, 6))
# plt.legend(loc='best')
#
# plt.show()
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = torch.autograd.Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=10,n_output=1)

optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
loss_f=torch.nn.MSELoss()

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
