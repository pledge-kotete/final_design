# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA #引入ICA函数

# #############################################################################


def ica_pre(data,wants_output):
    X=data
    # rows,cols=X.shape
    # average=np.mean(X,axis=0)
    # Compute ICA
    ica = FastICA(n_components=wants_output) #ICA计算核心代码，wants_output代表盲源个数
    S_ = ica.fit_transform(X)  # 对信号重构，得到分离后的信号
    A_ = ica.mixing_  # 计算混合矩阵


    # #############################################################################
    # 画出分离后源信号的散点波形图
    X_l = np.arange(0, len(S_))
    # a_l=S_[:,0]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'c','b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    plt.figure('ica')
    plt.suptitle("ica_"+str(wants_output))
    plot_num=np.ceil(wants_output/2)
    for i in range(wants_output):
        plt.subplot(plot_num,2,i+1)
        count=i
        print(count)
        plt.scatter(X_l,S_[:,i], c=colors[i],s=1,marker=".", alpha=0.5)
        plt.xlim(0, len(S_))
        plt.xlabel(str(i))
        plt.ylabel("amtitude")
    # plt.show()
    print("ica done")
    return S_