'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

# 计算香农熵
def calcShannonEnt(dataSet):
    # 计算整个数据集的长度
    numEntries = len(dataSet)
    # 用于存每个类别出现的次数（频率），需要计算比例
    labelCounts = {}
    # 遍历数据集中每一条数据，并纪录每条数据的label
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 如果labelCounts中没有这个label则将这个label(键)=>0(值)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 并将这个label的值进行+1
        labelCounts[currentLabel] += 1
    # 将香农熵先默认为0
    shannonEnt = 0.0
    # 计算每个label的熵的和
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 按照给定特征划分数据集（数据集，特征列，需要返回的特征）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 判断axis列的值是否为value
        if featVec[axis] == value:
            # [:axis]表示前axis列，即若axis为2，就是取featVec的前axis列
            reducedFeatVec = featVec[:axis]
            # [axis + 1:]表示从跳过axis + 1列，取接下来的数据
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 计算特征个数
    numFeatures = len(dataSet[0]) - 1
    # 计算香农值
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化应该选择的划分特征的信息增益和下标
    bestInfoGain = 0.0
    bestFeature = -1
    # 迭代所有特征
    for i in range(numFeatures):
        # 首先取出每一行example，然后将每一行中的第i列存入featList
        featList = [example[i] for example in dataSet]
        # 创建一个没有重复值的set(为了得到每个特征不同的值)
        uniqueVals = set(featList)
        # 初始化这个特征的信息增益
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer

# 返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 字典.iteritems()方法在需要迭代结果的时候使用最适合，而且它的工作效率非常的高。
    # 逆序排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的那个feature
    return sortedClassCount[0][0]

# 构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选择一个最合适的feature来进行划分
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 从label数组中删除这个label
    del (labels[bestFeat])
    # 获取这个label的不重复的所有值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
        输入：决策树，分类标签，测试数据
        输出：决策结果
        描述：跑决策树
        """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# 准备数据
filename='lenses.txt'
data=[]
with open(filename) as f:
    for line in f.readlines():
        arrLine=line.strip().split('\t')
        data.append(arrLine)
labels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(data,labels)
print(lensesTree)


# shanglong=calcShannonEnt(dataSet)
# retDataSet=splitDataSet(dataSet, 0, 1)
# print(str(retDataSet))
