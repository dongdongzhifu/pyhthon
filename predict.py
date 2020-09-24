import scipy.io.wavfile as wf
import keras
from keras.models import Sequential,load_model
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
def load_testdata():
    testwavs = []
    testlabels = []
    testlabsInd = []  # 测试集标签的名字   0：one   1：two

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
    testwavs = np.concatenate([x for x in testwavs])
    print('测试集总数',len(testwavs))
    testlabels = np.array(testlabels)
    return  (testwavs, testlabels),testlabsInd

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

def test():
    (testwavs, testlabels), testlabsInd = load_testdata()
    testwavs = testwavs.reshape(testwavs.shape[0], 13 * 99).astype('float32')
    testlabels = keras.utils.to_categorical(testlabels, 2)
    model = load_model('asr_model_weights.h5')
    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(testwavs, testlabels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])  # 准确度

if __name__ == '__main__':
    test()


