# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 12:31
# @Author  : GMell
# @Desc : ==============================================
# 绘制原始数据集
# ======================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

def sigmoid(inX):
    '''
    sigmoid函数计算
    :param inX: 入参
    :return: 西格玛（inx）
    '''
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    :param dataMatIn:二维np数组，本例中包含两个特征x,y,每行表示一个训练样本
    :param classLabels: 100*1的行向量
    :return:
    '''
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMat)
    alpha = 0.01
    # 梯度上升中移动的步长
    maxCycles = 500
    # 迭代的次数
    weight = np.ones((n, 1))
    # 进行矩阵运算，返回回归系数
    for k in range(maxCycles):
        # h为100*1的列向量
        h = sigmoid(dataMat*weight)
        error = (labelMat - h)
        weight = weight + alpha * dataMat.transpose()*error
    return weight


if __name__ == '__main__':
    scatterPlot()