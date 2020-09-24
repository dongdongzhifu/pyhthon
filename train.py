import scipy.io.wavfile as wf
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import numpy as np
import os

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

# 加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    wavs = []
    labels = []  # labels 和 testlabels 这里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    testwavs = []
    testlabels = []

    labsInd = []  # 训练集标签的名字   0：one   1：two
    testlabsInd = []  # 测试集标签的名字   0：one   1：two

    path = "E:\\ML_DL\\DL\\SpeechRecognition_en\\wav\\one\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path + i)
        # print(waveData)
        wavs.append(waveData)
        if ("one" in labsInd) == False:
            labsInd.append("one")
        labels.append(labsInd.index("one"))

    path = "E:\\ML_DL\\DL\\SpeechRecognition_en\\wav\\two\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path + i)
        wavs.append(waveData)
        if ("two" in labsInd) == False:
            labsInd.append("two")
        labels.append(labsInd.index("two"))

    # # 现在为了测试方便和快速直接写死，后面需要改成自动扫描文件夹和标签的形式
    path = "E:\\ML_DL\\DL\\SpeechRecognition_en\\test\\one\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path + i)
        testwavs.append(waveData)
        if ("one" in testlabsInd) == False:
            testlabsInd.append("one")
        testlabels.append(testlabsInd.index("one"))

    path = "E:\\ML_DL\\DL\\SpeechRecognition_en\\test\\two\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path + i)
        testwavs.append(waveData)
        if ("two" in testlabsInd) == False:
            testlabsInd.append("two")
        testlabels.append(testlabsInd.index("two"))
    wavs = np.concatenate([x for x in wavs])
    print('训练集总数',len(wavs))
    labels = np.array(labels)
    testwavs = np.concatenate([x for x in testwavs])
    print('训练集总数',len(testwavs))
    testlabels = np.array(testlabels)
    return (wavs, labels), (testwavs, testlabels), (labsInd, testlabsInd)
def get_wav_mfcc(wav_path):
    sample_rate, sigs = wf.read(wav_path)
    MFCC_data = mfcc(sigs, sample_rate).T
    MFCC_data = maxminnorm(MFCC_data)
    #print(MFCC_data.shape)
    # plt.matshow(MFCC_data, cmap='gist_rainbow')
    # plt.show()
    if len(MFCC_data[0]) != 99:
        MFCC_data = np.resize(MFCC_data,(13,99))
    MFCC_data = np.expand_dims(MFCC_data, axis=0)
    return MFCC_data

def trainData():
    (wavs, labels), (testwavs, testlabels), (labsInd, testlabsInd) = create_datasets()
    wavs = wavs.reshape(wavs.shape[0], 13*99).astype('float32')
    testwavs = testwavs.reshape(testwavs.shape[0], 13*99).astype('float32')
    print('--------------------------------------------------------')
    print(wavs.shape, "   ", labels.shape)
    print(testwavs.shape, "   ", testlabels.shape)
    print(labsInd, "  ", testlabsInd)
    print('--------------------------------------------------------')
    # 标签转换为独热码
    labels = keras.utils.to_categorical(labels, 2)
    testlabels = keras.utils.to_categorical(testlabels, 2)
    print(labels[0])  # 类似 [1. 0]
    print(testlabels[0])  # 类似 [0. 0]

    print(wavs.shape, "   ", labels.shape)
    print(testwavs.shape, "   ", testlabels.shape)

    # 构建模型
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(13*99,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print(model.summary())
    #  validation_data为验证集
    model.fit(
        wavs,
        labels,
        batch_size=124,
        epochs=300,
        verbose=2,
        validation_data=(
            testwavs,
            testlabels)
    )
    model.save('asr_model_weights.h5')  # 保存训练模型

if __name__ == '__main__':
    trainData()