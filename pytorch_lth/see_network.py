# coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_lth.visualize import make_dot

x = Variable(torch.randn(1,16,2400))#change 12 to the channel number of network input
class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden_1,n_hidden_2,n_hidden_3, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden_1)   # 隐藏层线性输出
        self.hidden_2 = torch.nn.Linear(n_hidden_1,n_hidden_2)
        self.hidden_3 = torch.nn.Linear(n_hidden_2,n_hidden_3)
        self.dropout =torch.nn.Dropout(p=0.5)
        self.hidden_4 = torch.nn.Linear(n_hidden_3, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden_1(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden_2(x))
        x = self.dropout(x)               # dropout 层
        x = F.relu(self.hidden_3(x))
        x = self.dropout(x)
        x = F.log_softmax(self.hidden_4(x),dim=1)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = Net(n_feature=2400, n_hidden_1=256,n_hidden_2=512, n_hidden_3=128,n_output=2) # 几个类别就几个 output

# print(AlexNet)# net 的结构

model=net
y = model(x)
g = make_dot(y)
g.view()