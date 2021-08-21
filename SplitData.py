"""
划分数据集的默认函数
author:zjp
"""

import scipy.io
import numpy as np
from sklearn.decomposition import PCA


def getTrainTest(trainPercent):
    """
    默认padSize:4, 均值方差归一化，降维到原先维度的1/3
    :param dataName: 数据集名称
    :param trainPercent: 训练集占比
    :return: trainX: shape=(tarinNum, 1, 9, 9, B//3) trainY: shape=(trainNum,1),
            testX: shape=(testNum, 1, 9, 9, B//3) testY: shape=(testNum,1),
    """
    rawData = scipy.io.loadmat("DataSet/Indian_pines/Indian_pines.mat")["indian_pines_corrected"]
    # rawData = scipy.io.loadmat("DataSet/Indian_Pines/Indian_Pines.mat")
    # print(rawData)
    groundTruth = scipy.io.loadmat("DataSet/Indian_pines/Indian_pines_gt.mat")["indian_pines_gt"]
    padSize = 4
    height, width, bands = rawData.shape
    tempx = np.reshape(rawData, [-1, bands])
    avgx = np.average(tempx, axis=0)
    stdx = np.std(tempx, axis=0)
    tempx = (tempx - avgx) / stdx - 1
    tempx = np.reshape(tempx, [height, width, -1])
    newx = np.reshape(tempx, (-1, tempx.shape[2]))
    pca = PCA(n_components=30, whiten=True)
    newx = pca.fit_transform(newx)
    newx = np.reshape(newx, (tempx.shape[0], tempx.shape[1], 30))

    padded = np.pad(newx, ((padSize, padSize), (padSize, padSize), (0, 0)),
                    'constant')
    bulkX, bulkY = [], []
    for i in range(height):
        for j in range(width):
            if groundTruth[i, j] != 0:
                bulkX.append(np.reshape(padded[i:i + 2 * padSize + 1, j:j + 2 * padSize + 1, :],
                                        [1, 2 * padSize + 1, 2 * padSize + 1, -1]))
                bulkY.append(groundTruth[i, j])
    dataX, dataY = np.array(bulkX), np.reshape(np.array(bulkY), [-1, 1]) - 1
    # 分离训练集
    trainNum = int(np.rint(trainPercent * dataX.shape[0]))
    state = np.random.get_state()
    np.random.shuffle(dataX)
    np.random.set_state(state)
    np.random.shuffle(dataY)
    trainX, trainY = dataX[0:trainNum, :, :, :, :], dataY[0:trainNum, :]
    # totalnumPerclass = np.array([np.sum(dataY == i) for i in range(0, np.max(dataY) + 1)])
    # trainnumPerclass = np.array([np.sum(trainY == i) for i in range(0, np.max(dataY) + 1)])
    return trainX, trainY, dataX[trainNum:, :, :, :, :], dataY[trainNum:, :]
