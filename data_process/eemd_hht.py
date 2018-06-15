# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pyeemd # 整体经验模态分解包
import pyemd #经验模态分解包（效果差于eemd）
import pyhht #直接对数据进行希尔伯特黄变换，但是由于并未用整体经验模态分解法，因此在此不考虑
from pyeemd.utils import plot_imfs
from scipy.fftpack import hilbert #引入scipy中的hilbert变换函数
import pyhht.utils as hhtu

hilbert_out=[]
def my_hht(data):
    X=np.sum(data,axis=1)
    X=X/len(data[0])
    plt.figure("hht_8")
    # X_0=hhtu.boundary_conditions(X,np.arange(len(X)))

    imfs=pyeemd.ceemdan(X) # EEMD处理部分，可得到imf（本证模态函数）,所有参数均使用默认值即可,ceemdan是一种更优于eemd的数据处理方法，
    #其增添的噪声数据会自动选择当前情况下最好的噪声进行运算，但是运算时间会略微加长

    # 画出ceemdan运算后所有imf的波形图
    # plot_imfs(imfs,plot_splines=False)

    #画出hht后数据的散点图
    plot_num=np.ceil((len(imfs)-1)/2)
    for i in range(len(imfs)-1):
        hilbert_1=hilbert(imfs[i])
        hilbert_out.append(hilbert(imfs[i]))
        plt.subplot(plot_num, 2, i + 1)
        count = i
        print("hht_" + str(count))
        plt.plot(hilbert_1,label="hht_"+str(count))
        plt.legend(loc='best', frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=3)
        plt.ylabel("amtitude")
    plt.show()
    return hilbert_out
