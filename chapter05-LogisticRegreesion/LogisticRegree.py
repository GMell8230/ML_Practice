# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 12:31
# @Author  : GMell
# @Desc : ==============================================
# 绘制原始数据集并通过梯度上升法计算最优w，并绘制决策边界
# ======================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random
from matplotlib.patches import Rectangle

def loadData():
    '''
    加载数据集
    :return:数据矩阵dataMat, 标签矩阵lableMat
    '''
    dataMat = []; lableMat = []
    fr = open('DataSet/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为了计算方便，将第0列置为1.0，第二列对应x1,第三列对应x2
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))
    return dataMat, lableMat

def sigmoid(inX):
    '''
    sigmoid函数计算
    :param inX: 入参
    :return: 西格玛（inx）
    '''
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    :param dataMatIn:二维np数组，本例中包含两个特征x,y,每行表示一个训练样本
    :param classLabels: 100*1的行向量
    :return:
    '''
    dataMat = np.mat(dataMatIn)
    # transpose矩阵的转置
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    # 梯度上升中移动的步长
    alpha = 0.01
    # 迭代的次数
    maxCycles = 500
    weight = np.ones((n, 1))

    # 进行矩阵运算，返回回归系数
    for k in range(maxCycles):
        # h为100*1的列向量
        h = sigmoid(dataMat*weight)
        error = (labelMat - h)
        weight = weight + alpha * dataMat.transpose()*error
    return weight

def stocGradAscent0(dataMatrix, classLabels):
    '''
    梯度上升算法
    :param dataMatrix:数据集
    :param classLabels:标签
    :return:回归系数
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 10):
    '''
    改进的梯度上升算法
    :param dataMatrix:数据集
    :param classLabels:标签
    :param numIter:迭代次数
    :return:回归系数
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    # 设置算法的迭代次数
    for j in range(numIter):
        # 从m行数据中任选一行作为样本
        dataIndex = list(range(m))
        for i in range(m):
            # 将步长在每次进行迭代时候都进行调整，以削弱迭代过程中回归系数的波动
            alpha = 4/(1.0+i+j)+0.01
            # 随机选择样本更新回归系数，减少周期性的波动
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    '''
    绘制决策的边界
    :param weight:回归系数
    :return: 根据回归划分的图像
    '''
    # weight.getA(): numpy.matrix.getA Return self as an ndarray object.
    # weights = weight.getA()
    dataMat, labelMat = loadData()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    # 1行1列第一个图
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, color='red', marker = 's', label = 'point1')
    ax.scatter(xcord2, ycord2, s=30, color='green',label = 'point2')
    x = np.arange(-5.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y, label = 'dicision border')
    plt.xlabel('X1');plt.ylabel('X2');
    plt.grid()
    plt.legend()
    plt.show()

def scatterPlot():
    '''
    绘制数据集的散点图
    :return:散点图
    '''
    dataMat, lableMat = loadData()
    dataArr = np.array(dataMat)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    n = np.shape(dataArr)[0]
    for i in range(n):
        if int(lableMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure(2);
    ax = fig.add_subplot(111)
    #ax.scatter(xcord,ycord, c=colors, s=markers)
    type1 = ax.scatter(xcord1, ycord1, c='red', s=30, label='class 0', marker='s')
    type2 = ax.scatter(xcord2, ycord2, c='green', label='class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'upper right')
    plt.show()
    print("绘制成功")

# implement
dataMatIn, classLabels = loadData()
weight = stocGradAscent1(np.array(dataMatIn), classLabels)
print(weight)
plotBestFit(weight)