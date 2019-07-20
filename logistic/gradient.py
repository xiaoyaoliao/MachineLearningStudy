# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:06:53 2018
@author: Administrator
"""

from numpy import *


def loadData(filename):
    '''
    '''
    datamat = [];
    labelmat = []
    with open(filename) as fr:
        for line in fr.readlines():
            # 以tab切割，并去掉两头的空格
            line_arr = line.strip().split()
            # 构建数据集，前面加上一列1
            datamat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            # jisuan x0, x1,x2. x0wei 1
            # 构建结果数组
            labelmat.append(int(line_arr[2]))
    # 返回这两个数组
    return datamat, labelmat

# 定义logistic回归的sigmoid函数
def sigmoid(inp):
    return 1.0 / (1 + exp(-inp))

def stocGradAscent0(datamat,labels,numIter=150):
    m,n=shape(datamat)
    weights=ones(n)
    for j in range(numIter):
        # 统计数据集有多少条数据，注：range()不能del,所以要转换成list
        dataIndex=list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本
            randIndex=int(random.uniform(0,len(dataIndex)))
            # 选择随机选取的一个样本，计算h
            h=sigmoid(sum(datamat[randIndex]*weights))
            # 计算误差
            error=labels[randIndex]-h
            # 更新回归系数
            weights=weights+alpha*error*datamat[randIndex]
            # 删除已经使用的样本
            del (dataIndex[randIndex])
    return weights

def Grad_descent(datamat, labels):
    # mat()是转换成python数据能够计算的格式
    data = mat(datamat)
    # transpose是将矩阵进行转置
    label = mat(labels).transpose()
    # m n分别等于矩阵datamat矩阵的行与列
    m, n = shape(datamat)
    # 初始化学习率为0.001，最大迭代次数为500
    alpha = 0.001;
    max_iter = 500
    # 存储每个theta的值
    weights = ones((n, 1))
    for k in range(max_iter):
        #矩阵相乘
        z = dot(datamat, weights)
        y_pred = sigmoid(z)
        error = (label - y_pred)
        # grad(x) = (y - f(x)) * x'
        weights = weights + alpha * data.transpose() * error
    return weights

# 画出决策边界
import matplotlib.pyplot as plt

def plot_fit(data, labelMat, weights):
    # 训练集
    dataArr = array(data)
    # 读取行数
    n = shape(dataArr)[0]

    x_cord1 = [];y_cord1 = []
    x_cord2 = [];y_cord2 = []
    # 遍历训练集，如果该example为1则将坐标添加到 x_cord1 = [];y_cord1 = []
    # 如果该example为0则将坐标添加到x_cord2 = [];y_cord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            x_cord1.append(dataArr[i, 1])
            y_cord1.append(dataArr[i, 2])
        else:
            x_cord2.append(dataArr[i, 1])
            y_cord2.append(dataArr[i, 2])
    #绘制各个不同点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    # 绘制决策边界
    x = arange(-3.0, 3.0, 0.1)
    # 拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2, x0为1,x1为x, x2为y,故有
    y = ((-weights[0] - weights[1] * x) / weights[2]).transpose()
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# def plotWeights(weights_array1,weights_array2):
#     #设置汉字格式
#     font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
#     #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
#     #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
#     fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
#     x1 = np.arange(0, len(weights_array1), 1)
#     #绘制w0与迭代次数的关系
#     axs[0][0].plot(x1,weights_array1[:,0])
#     axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
#     axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
#     plt.setp(axs0_title_text, size=20, weight='bold', color='black')
#     plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
#     #绘制w1与迭代次数的关系
#     axs[1][0].plot(x1,weights_array1[:,1])
#     axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
#     plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
#     #绘制w2与迭代次数的关系
#     axs[2][0].plot(x1,weights_array1[:,2])
#     axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
#     axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
#     plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
#     plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
#
#
#     x2 = np.arange(0, len(weights_array2), 1)
#     #绘制w0与迭代次数的关系
#     axs[0][1].plot(x2,weights_array2[:,0])
#     axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
#     axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
#     plt.setp(axs0_title_text, size=20, weight='bold', color='black')
#     plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
#     #绘制w1与迭代次数的关系
#     axs[1][1].plot(x2,weights_array2[:,1])
#     axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
#     plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
#     #绘制w2与迭代次数的关系
#     axs[2][1].plot(x2,weights_array2[:,2])
#     axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
#     axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
#     plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
#     plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
#
#     plt.show()
#
# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)
#
#     weights2,weights_array2 = gradAscent(dataMat, labelMat)
#     plotWeights(weights_array1, weights_array2)


filename='testSet.txt'
datemat,datalable=loadData(filename)
weights=stocGradAscent0(array(datemat),datalable)
plot_fit(datemat,datalable,weights)