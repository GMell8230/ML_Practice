# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 15:46
# @Author  : GMell
# @Desc : ==============================================
# 进行决策树的构造，绘制及结果测试 ===
# ======================================================

from math import log
import operator

def creatDataSet():
    dataSet = [[1, 1, 'yes'],
             [1, 1, 'yes'],
             [1, 0, 'no'],
             [0, 1, 'no'],
             [0, 1, 'no'],]
    labels = ['no sufacing', 'flippers']
    return dataSet, labels

def calculateShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵,度量数据集的无序程度
    :param dataSet: 原始数据集
    :return: shannonEnt 香农熵
    '''
    numEntrie = len(dataSet)
    #数据集中的样本总数
    labelCount = {}    #标签字典
    for featureVec in dataSet:
    #遍历数据集，将所有的标签分类记入字典
        currentLabel = featureVec[-1]
        if currentLabel not in labelCount.keys():
            #字典中若无此标签，则将此标签加入字典并初始化为0
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
    #统计每个标签在总体分类中出现的概率，并计算香农熵
        prob = float(labelCount[key])/numEntrie
        #每个标签出现的概率
        shannonEnt -= prob * log(prob,2)
        # 香农熵 = -Σ(1,n)p(xi)*log(p(xi),2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 原始数据集
    :param axis: 给定特征的维度（本例为第几个特征）
    :param value: 给定特征的值
    :return:retDataSet：以axis维特征给定value值后，划分的数据集
    '''
    retDataSet = []
    for featVec in dataSet:
    #遍历dataSet
        if featVec[axis] == value:
        #若本条样本的第axis维特征等于给定特征值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            #记录下除axis维特征前后的特征值
            retDataSet.append(reducedFeatVec)
            #追加进入划分后数据集
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseShannoEnt = calculateShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featureslist = [example[i] for example in dataSet]
        uniqueVals = set(featureslist)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy = prob * calculateShannonEnt(subDataSet)
        infoGain = baseShannoEnt - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    数据集处理完所有属性，但是类标签仍然不唯一时，此时通过投票决定定义叶子节点
    :param classList: 分类名称列表
    :return: 分类中投票数最多的标签
    '''
    classCount = {}  #定义字典统计各属性投票的次数
    for vote in classList:
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    #operator.itemgetter(1)根据字典的第二维进行排序
    return sortedClassCount[0][0]

def creatTree(dataSet, labels):
    '''
    创建决策树
    :param dataSet:数据集
    :param labels: 标签
    :return: 决策树
    '''
    classList = [example[-1] for example in dataSet]   #数据集的标签列表

    if classList.count(classList[0]) == len(classList):
        #递归第一个出口：所有的类标签相同
        return classList[0]
    if len(dataSet[0]) == 1:
        #递归第二个出口：所有特征已被访问完毕，但仍然不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)

    # 当前数据集的最好特征维数
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = labels[bestFeat]
    # 通过字典建立树的结构
    myTree = {bestFeatLable:{}}
    # 特征已经入树结点，在标签中删除此特征标签
    del(labels[bestFeat])
    #得到列表中，第i维特征的所有属性值
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    for val in uniqueVals:
        #遍历所有属性，递归调用creatTree，最后返回myTree
        subLabels = labels[:]
        #复制labels中的标签，防止在递归中篡改
        myTree[bestFeatLable][val] = creatTree(splitDataSet(dataSet,bestFeat,val), subLabels)

    return myTree

def classfy(inputTree, featLabels, testVec):
    '''
    递归执行数据分类(给定决策树下，输入向量，输出分类结果)
    :param inputTree:决策树
    :param featLabels:特征的标签
    :param testVec:测试向量
    :return:分类结果
    '''
    firstStr = list(inputTree.keys())[0]
    #取决策树字典中的第一位，即标签值字符串
    secondDict = inputTree[firstStr]
    #字典中key 对应的value 即该结点下的数据
    featIndex = featLabels.index(firstStr)
    #找到该特征对应的索引
    for key in secondDict.keys():
    #遍历节点下key
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
            #若key对应的value为字典类型，则继续递归分类
                classLabel = classfy(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
            #得到分类结果
    return classLabel

def storeTree(inputTree, filename):
    '''
    将决策树通过pickle模块存储
    :param inputTree: 决策树
    :param filename: 存储路径
    :return:
    '''
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    读取决策树
    :param filename:决策树的存储路径
    :return: 控制台输出
    '''
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)



def main():
    myData, labels = creatDataSet()
    # 计算香农熵
    # shannoEnt = calculateShannonEnt(myData)
    # print(myData)
    # print(shannoEnt)

    # 测试选取同一特征不同特征值划分的两个数据集
    # split1 = splitDataSet(myData, 0, 1)
    # split2 = splitDataSet(myData, 0, 0)
    # print(split1)
    # print(split2)

    myTree = creatTree(myData,labels)
    print(myTree)

    myData, labels = creatDataSet()

    res1 = classfy(myTree, labels, [1,0])
    print(res1)
    res2 = classfy(myTree, labels, [1,1])
    print(res2)

    storeTree(myTree, 'TreeClassifierSavedfile.txt')
    print(grabTree('TreeClassifierSavedfile.txt'))

if __name__ == '__main__':
    main()