# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 14:25
# @Author  : GMell
# @Desc : ==============================================
# 通过朴素贝叶斯分类器实现文本分类
# ======================================================
import numpy as np
import math


def loadDataSet():
    '''
    加载实验样本
    :return:postingList,classVec
    '''
    # 词条切分后的留言
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so',
                    'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupic',
                    'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                    'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless',
                   'dog', 'food', 'stupid']]
   # 手工标注的文本类别:1 代表侮辱性文字 0 代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]

    return  postingList, classVec

def createVocabList(dataSet):
    '''
    创建数据集的词汇本
    :param dataSet: 数据集
    :return: 数据集中词汇的列表
    '''
    #创建词汇的空集
    vocabSet = set([])
    for document in dataSet:
        #创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    词集模型
    :param vocabList: 创建的词汇表
    :param inputSet: 文本数据集
    :return: 文本数据集的向量值
    '''
    #创建len长度的0向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print('this word: %s is not in my Vocabulary' %word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    '''
    词带模型
    :param vocabList: 创建的词汇表
    :param inputSet: 文本数据集
    :return: 文本数据集的向量值
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trianNBayes0(trainMatrix, trainCategory):
    '''
    native-Bayes训练函数 计算侮辱性和普通语句下词出现的概率
    :param trainMatrix:文档的矩阵
    :param trainCategory: 每篇文档类别标签构成的向量
    :return:p0Vect, p1Vect, p_Abusive
    '''
    # 训练矩阵中的语句数量
    numTrainDocs = len(trainMatrix)
    # 矩阵的列数，即语句的长度，词的个数
    numWords = len(trainMatrix[0])
    # 计算语句属于侮辱性语句的概率，trainCategory中是侮辱性的语句对应向量值为1
    p_Abusive = sum(trainCategory) / float(numTrainDocs)
    # p0：正常语句初始化，p1：侮辱性语句初始化
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    # 遍历整个文档，统计
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect, p1Vect, p_Abusive

def classifyNBayes0(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    朴素贝叶斯分类函数
    :param vec2Classify:要执行分类的向量
    :param p0Vec: 在正常语句中，各个单词出现的概率
    :param p1Vec: 在侮辱性语句中，各个单词出现的概率
    :param pClass1: 侮辱性语句在文本中出现的概率
    :return:
    '''
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1 )
    if p1 > p0:
        return 1
    else:
        return 0




def main():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    myVocabList.sort()
    print(myVocabList)
    # wordsVec = setOfWords2Vec(myVocabList,listPosts[0])

    trainMat = []
    for postInDoc in listPosts:
        #填充trianMat，将每句话中的词汇向量化加入trainMat
        trainMat.append(setOfWords2Vec(myVocabList,postInDoc))

    p0V, p1V, pAb = trianNBayes0(trainMat, listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "Entry1 class as :", classifyNBayes0(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "Entry2 class as :", classifyNBayes0(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    main()




















