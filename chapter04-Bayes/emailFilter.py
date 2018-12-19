# -*- coding: utf-8 -*-
# @Time    : 2018/12/19 15:47
# @Author  : GMell
# @Desc : ==============================================
# 使用朴素贝叶斯分类器实现垃圾邮件的过滤
# ①准备数据，将语句切割成为词向量
# ②导入测试数据进行分类测试
# ======================================================
import random
import re
import nativeBayes as nbayes
import os
import numpy as np


def sentence2wrod(sentence):
    '''
    将语句切分成词列表
    :param sentence:输入语句
    :return: 词列表
    '''
    # wordList = sentence.split()
    regEx = re.compile('\\W')
    listOfTokens = regEx.split(sentence)
    listOfTokens = [tok.lower() for tok in listOfTokens if len(tok)>0]
    return listOfTokens

def spamTest(p0dataSetpath,p1dataSetpath):
    '''
    垃圾邮件的检测测试
    :param p0dataSetpath:正常邮件数据集的路径
    :param p1dataSetpath:垃圾邮件数据集的路径
    :return:
    '''
    docList = []; classList = []; fullList = []
    normalCount = len([name for name in os.listdir(p0dataSetpath)])
    spamCount = len([name for name in os.listdir(p1dataSetpath)])

    # 遍历每个正常数据目录下的数据集
    for i in range(1,26):
        wordList = sentence2wrod(open('email/ham/%d.txt' %i).read())
        #追加第i个文本：句子
        docList.append(wordList)
        #扩展词汇容量：词语
        fullList.extend(wordList)
        classList.append(1)
    for i in range(1,26):
        wordList = sentence2wrod(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullList.extend(wordList)
        classList.append(0)
    vocabList = nbayes.createVocabList(docList)
    trainingSet = list(range(normalCount+spamCount)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(nbayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = nbayes.trianNBayes0(np.array(trainMat),np.array(trainClass))
    errCount = 0
    for docIndex in testSet:
        wordVector = nbayes.setOfWords2Vec(vocabList, docList[docIndex])
        if nbayes.classifyNBayes0(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errCount += 1
    print('the err rate is: ' ,float(errCount)/len(testSet))
def main():
    # mySentence = 'This book is the best book on python or M.L. I have \
    #               ever laid eyes on.'
    # wordList = sentence2wrod(mySentence)
    # print(wordList)
    p0dataSetpath = 'email/ham'
    p1dataSetpath = 'email/spam'
    spamTest(p0dataSetpath, p1dataSetpath)

if __name__ == '__main__':
    main()