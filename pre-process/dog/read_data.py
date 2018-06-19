# coding=utf-8
# 所有的函数都可以用Ctrl+b的快捷键直接定位
import numpy as np
import scipy.io as scio
from data_process import ica_process, eemd_hht,wavelet

rootpath="/media/eelab/data/kaggle/"
# txt_path=rootpath+"file.txt"
txt_path=rootpath+"dog_test.txt" #狗的数据文件
dog_filename=[] #含有狗的数据的文件名

# 读入list文件
with open(txt_path) as f:
    lines=f.readlines()
for line in lines:
    kinds=line.strip("\n").split("_")
    whole_path=rootpath+line.strip()
    dog_filename.append(whole_path)

#dogs part
for filepath in dog_filename:
    dataFile=filepath

    #数据读取部分（此部分可直接使用），最重要的为data与label
    data_0=scio.loadmat(dataFile) #读入原始dict文件
    keys=dataFile.split(".")[0].split("_")
    count=int(keys[-1])
    final_key=keys[-3]+"_"+keys[-2]+"_"+str(count)
    data_1=data_0[final_key]
    print(final_key)
    datalist=data_1.tolist()
    data_2=datalist[0][0]
    data=data_2[0].tolist()
    X=np.arange(0,len(data[00]))
    data_length=data_2[1].tolist()[0][0]
    sampling_frequency=data_2[2].tolist()[0][0]
    channels=data_2[3].tolist()[0]
    if(keys[-3])=="test":
        continue
    elif(keys[-3])=="preictal":
        sequence = data_2[4].tolist()[0][0]
        label="1"
    elif(keys[-3])=="interictal":
        sequence = data_2[4].tolist()[0][0]
        label = "0"
    data=np.array(data)
    data_T=data.T

    # 数据预处理
    ica_out_8= ica_process.ica_pre(data_T, 8) # 将所有原始信号经过ICA转变成8个独立源信号
    ica_out_12= ica_process.ica_pre(data_T, 12) # 将所有原始信号经过ICA转变成12个独立源信号
    # wavelet_out_8_haar_1=wavelet.my_wavelet(ica_out_8,'haar') # 针对上面得到的8个独立源信号，使用haar小波进行小波变换
    # wavelet_out_8_db2_1=wavelet.my_wavelet(ica_out_8,'db2') # 针对上面得到的8个独立源信号，使用db2小波进行小波变换
    # wavelet_out_12_haar_1=wavelet.my_wavelet(ica_out_12,'haar') # 针对上面得到的12个独立源信号，使用haar小波进行小波变换
    # wavelet_out_12_db2_1=wavelet.my_wavelet(ica_out_12,'db2') # 针对上面得到的12个独立源信号，使用db2小波进行小波变换
    # hht_8_out=eemd_hht.my_hht(ica_out_8) # 针对上面得到的8个独立成分源信号，对其进行希尔伯特黄变换
    hht_12_out= eemd_hht.my_hht(ica_out_12) # 针对上面得到的12个独立成分源信号，对其进行希尔伯特黄变换

    # 画出原始数据的散点图
    # plt.figure('origin')
    # plt.suptitle(final_key)
    # for i in range(len(channels)):
    #     plt.subplot(8, 2, i + 1)
    #     count = i
    #     print(count)
    #     plt.scatter(X, np.array(data[count]), c="c", s=1, marker=".", alpha=0.5, label=channels[count].tolist())
    #     plt.xlim(0, len(data[count]))
    #     plt.legend(loc='best', frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=3)
    #     plt.ylabel("amtitude")
    # plt.show()

