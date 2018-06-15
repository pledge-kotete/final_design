# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

rootpath="/media/eelab/data/kaggle/"
txt_path=rootpath+"file.txt"
# txt_path=rootpath+"test.txt"
dog_filename=[] #all dogs have 16 electrode
human_15_filename=[] #human 1 has 15 electrode
human_24_filename=[] #human 2 has 24 electrode
with open(txt_path) as f:
    lines=f.readlines()
for line in lines:
    kinds=line.strip("\n").split("_")
    whole_path=rootpath+line.strip()
    if kinds[0]=="Dog":
        dog_filename.append(whole_path)
    elif kinds[1]=="1":
        human_15_filename.append(whole_path)
    elif kinds[1]=="2":
        human_24_filename.append(whole_path)

#dogs part
for filepath in dog_filename:
    dataFile=filepath
    data_0=scio.loadmat(dataFile)
    keys=dataFile.split(".")[0].split("_")
    count=int(keys[-1])
    final_key=keys[-3]+"_"+keys[-2]+"_"+str(count)
    data_1=data_0[final_key]
    datalist=data_1.tolist()
    data_2=datalist[0][0]
    data=data_2[0].tolist()
    X=np.arange(0,len(data[00]))
    plt.scatter(X,np.array(data[00]),s=1,alpha=0.5)
    plt.xlim(0,len(data[00]))
    plt.show()
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

    print(3)

#human 1 part
for filepath in human_15_filename:
    dataFile=filepath
    data_0=scio.loadmat(dataFile)
    keys=dataFile.split(".")[0].split("_")
    count=int(keys[-1])
    final_key=keys[-3]+"_"+keys[-2]+"_"+str(count)
    data_1=data_0[final_key]
    datalist=data_1.tolist()
    data_2=datalist[0][0]
    data=data_2[0].tolist()
    data_length=data_2[1].tolist()[0][0]
    sampling_frequency=data_2[2].tolist()[0][0]
    channels=data_2[3].tolist()[0]
    if (keys[-3]) == "test":
        continue
    elif (keys[-3]) == "preictal":
        sequence = data_2[4].tolist()[0][0]
        label = "1"
    elif (keys[-3]) == "interictal":
        sequence = data_2[4].tolist()[0][0]
        label = "0"

#human 2 part
for filepath in human_24_filename:
    dataFile=filepath
    data_0=scio.loadmat(dataFile)
    keys=dataFile.split(".")[0].split("_")
    count=int(keys[-1])
    final_key=keys[-3]+"_"+keys[-2]+"_"+str(count)
    data_1=data_0[final_key]
    datalist=data_1.tolist()
    data_2=datalist[0][0]
    data=data_2[0].tolist()
    data_length=data_2[1].tolist()[0][0]
    sampling_frequency=data_2[2].tolist()[0][0]
    channels=data_2[3].tolist()[0]
    if (keys[-3]) == "test":
        continue
    elif (keys[-3]) == "preictal":
        sequence = data_2[4].tolist()[0][0]
        label = "1"
    elif (keys[-3]) == "interictal":
        sequence = data_2[4].tolist()[0][0]
        label = "0"