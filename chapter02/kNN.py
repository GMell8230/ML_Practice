import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    '''
    第一个初始数据集
    :return: 标签和点集
    '''
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    KNN-分类
    :param inX: 输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 参数即选择邻近的数目
    :return: sortedClassCount[0][0] 最邻近的标签
    '''
    dataSetSize = dataSet.shape[0]
    #shape[0]返回数据集的行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #numpy.tile(A, reps): Construct an array by repeating A the number of times given by reps
    #计算出inX 与数据集中每个点之间的距离
    sqDiffMat = diffMat**2
    # diffMat**2 返回diffMat的2次幂 计算出(x^2, y^2)
    sqDistances = sqDiffMat.sum(axis=1)
    # sqDistances = x^2 + y^2
    distances = sqDistances**0.5
    #  distances = 开方sqDistances
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #classCount.get(voteIlabel,0)字典的用法寻找voteIlabel元素，并返回voteIlabel对应的值，默认值为0
    sortedClassCount = sorted(classCount.items(),
            key=operator.itemgetter(1), reverse=True)
    #将classCount字典分解为元祖列表，用itemgetter方法，按照第二个元素次序进行排序
    #classCount.items()返回迭代器
    #operator 模块提供的itemgetter函数用于获取对象的哪些维的数据
    #reverse 排序规则，reverse = True 降序 ， reverse = False 升序（
    return sortedClassCount[0][0]

def file2matrix(filename):
    '''
    将txt文本文件转为Numpy的解析程序
    :param filename: 文件名
    :return: 训练样本矩阵returnMat,类标签向量classLableVector
    '''
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)

    returnMat = np.zeros((numberOfLines,3))
    classLableVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        #strip('a')去除字符串中的包含的字符'a' 空则去除首尾的空格
        listFromLine = line.split('\t')
        #split()通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3]
        #将returnMat的第index行赋值为listFromLine 0~3列的数据
        classLableVector.append(int(listFromLine[-1]))
        #通过负索引将listFromLine最后一列数据存储到classLableVector
        index += 1
    return returnMat,classLableVector

def autoNorm(dataSet):
    '''
    过大过小的特征值会影响邻近距离的计算，需要进行归一化处理
    :param dataSet:原始数据
    :return: 归一化后的数据集
    '''
    minVal = dataSet.min(0)
    #每列的最小值
    maxVal = dataSet.max(0)
    #每列的最大值
    valRange = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #返回矩阵第一维的数值 即：行
    normDataSet = dataSet - np.tile(minVal,(m,1))
    #np.tile(minVal,(m,1)) 列不变，行加m倍
    normDataSet = normDataSet/ np.tile(valRange, (m, 1))

    return normDataSet, valRange, minVal

def datingClassTest():
    '''
    算法有效性测试
    :return: 算法有效性的比率
    '''
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('dataSet//datingTestSet2.txt')
    normMat, valRange, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #设定测试向量的数量
    errCount = 0.0
    for i in range(numTestVecs):
        classifierRes = classify0(normMat[i,:], normMat[numTestVecs:m],
                                  datingLabels[numTestVecs:m], 3)
        #输入第i个样本，通过kNN与源数据集分类，近邻值k为3，返回分类的标签值
        print ("the classifier came back with: %d , the real answer is: %d "
                %(classifierRes,datingLabels[i]))
        #若分类标签与原标签不相等，错误值+1
        if classifierRes != datingLabels[i]:
            errCount += 1
    print("total err is: %f" %(errCount/float(numTestVecs)))

def classifyPerson():
    '''
    约会网站预测函数，通过控制台交互，判定是否为目标人群
    :return: 讨厌、喜欢、很喜欢
    '''
    resultList = ['not at all','in small doses', 'in large doses']
    gameTime = 1.0
    flyingMiles = 1.0
    iceCreamRatio = 1.0
    while (gameTime > 0 & flyingMiles > 0 & iceCreamRatio > 0):
        gameTime = float(input(
            "percentage of time spent playing games?(0~100)"))
        flyingMiles = float(input(
            "ferquent fileer miles earned per year?"))
        iceCreamRatio = float(input(
            "liters of iceCream consumed per year?"))
        datingDataMat , datingLabels = file2matrix('dataSet\datingTestSet2.txt')
        normMat, valRange, minVals = autoNorm(datingDataMat)
        inputArr = np.array([flyingMiles, gameTime, iceCreamRatio])
        classifierRes = classify0((inputArr-minVals)/valRange, normMat, datingLabels, 3)
        print("your feeling of this person: ", resultList[classifierRes - 1])


def main():

    # 展示kNN算法
    # dataSet, labels = createDataSet()
    # inX = [0,0]
    # k = 3
    # res = classify0(inX, dataSet, labels, k)
    # print(res)

    #获取datingTestSet2数据
    #datingDataMat, datingLables = file2matrix('E:\PyCharmWorkplace\MLforWebapp\chapter02\dataSet\datingTestSet2.txt')

    # 使用matplotlib绘制datingTestSet2散点图

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for i in range(len(datingLables)):
    #     gameTime = datingDataMat[i:i+1, 1]
    #     iceCreamEat = datingDataMat[i:i+1, 2]
    #     flyingTime = datingDataMat[i:i+1, 0]
    #
    #     if datingLables[i] == 1:
    #         ax.scatter(gameTime, flyingTime, color='red')
    #     if datingLables[i] == 2:
    #         ax.scatter(gameTime, flyingTime, color = 'black')
    #     if datingLables[i] == 3:
    #         ax.scatter(gameTime, flyingTime, color='green')
    #
    # plt.show()

    # 归一化数值
    # normMat , valRange , minVal = autoNorm(datingDataMat)
    # print(normMat)
    # print(valRange)
    # print(minVal)

    #测试分类结果
    # datingClassTest()

    #输入数据并测试分类结果
    classifyPerson()

if __name__ == '__main__':
    main()
