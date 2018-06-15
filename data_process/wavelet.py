# coding=utf-8

import numpy as np
import pywt #小波变换原始库文件
import matplotlib.pyplot as plt

# w0=pywt.Wavelet("haar")
# w1=pywt.Wavelet("db1")
# w2=pywt.Wavelet("db2")
# w3=pywt.Wavelet("db3")

def my_wavelet(data,wavelet_name):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'c', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    plt.figure('wavelet_'+wavelet_name+"_"+str(len(data[0])))
    plt.suptitle('wavelet_'+wavelet_name+"_"+str(len(data[0])))
    plot_num = np.ceil(len(data[0])/2)
    cA_all=[]
    cD_all=[]
    for i in range(len(data[0])):
        X=data[:,i]
        cA,cD=pywt.dwt(X,wavelet_name) #小波变换模块
        X_l = np.arange(0, len(cA))
        cA_all.append(cA) #ca 为dwt后的低频信号
        cD_all.append(cD) #cd 为dwt后的高频信号

        # ca part
        # 画变换后信号的散点波形图
        plt.subplot(plot_num, 2, i+1 )
        count = i
        print("cA_"+str(count))
        plt.scatter(X_l, cA, c=colors[i], s=1, marker=".", alpha=0.5,label="cA_"+str(count))
        plt.xlim(0, len(X_l))
        plt.ylim(-0.05,0.05)
        plt.legend(loc='best', frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=3)
        plt.ylabel("amtitude")

        # cd part
        # plt.subplot(plot_num, 2, 2*i +2)
        # count = i
        # print("cD_" + str(count))
        # plt.scatter(X_l, cD, c=colors[i], s=1, marker=".", alpha=0.5,label="cA_"+str(count))
        # plt.xlim(0, len(X_l))
        # plt.ylim(-0.001,0.001)
        # plt.legend(loc='best', frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=3)
        # plt.ylabel("amtitude")
    # plt.show()
    # 由于脑电信号主要为低频信号，因此在此保留ca_all的所有数据
    return cA_all
