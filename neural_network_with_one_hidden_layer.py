# 构建具有单隐藏层的2类分类神经网络。
# 使用具有非线性激活功能激活函数，例如tanh。
# 计算交叉熵损失（损失函数）。
# 实现向前和向后传播。

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
# from week_3 import *

X, Y = load_planar_dataset()

print(X.shape)  # (2, 400)
print(Y.shape)  # (1, 400)

# 可视化数据集
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()

''''
def sigmoid(z):
    return 1/(1+np.exp(-z))
'''


def main():

    num_iterations = 1000000
    alpha = 0.4  # learning_rate

    m = Y.shape[1]  # the number of train_set
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 4

    """
        参数：
            n_x - 输入节点的数量
            n_h - 隐藏层节点的数量
            n_y - 输出层节点的数量

        返回：
            parameters - 包含参数的字典：
                W1 - 权重矩阵,维度为（n_h，n_x）
                b1 - 偏向量，维度为（n_h，1）
                W2 - 权重矩阵，维度为（n_y，n_h）
                b2 - 偏向量，维度为（n_y，1）

    """
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x)*0.01  # 防止权重矩阵数值过大
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros(shape=(n_y, 1))

    '''
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    '''

    # start training
    for i in range(num_iterations):

        # forward propagate
        z1 = np.dot(w1, X) + b1
        A1 = np.tanh(z1)
        z2 = np.dot(w2, A1) + b2
        A2 = sigmoid(z2)

        cost = np.multiply(np.log(A2), Y)+np.multiply((1 - Y), np.log(1 - A2))
        cost = -np.sum(cost)/m
        cost = float(np.squeeze(cost))

        # back propagate
        dz2 = A2 - Y
        dw2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.multiply(np.dot(w2.T, dz2), 1-np.power(A1, 2))  # np.power(x,y)==x**y
        dw1 = (1 / m) * (np.dot(dz1, X.T))
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        w1 = w1 - alpha * dw1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dw2
        b2 = b2 - alpha * db2

        if i % 100000 == 0:
            print("the number of iterations: ", i, " the error: ", str(cost))

    # prediction
    z1 = np.dot(w1, X) + b1
    A1 = np.tanh(z1)
    z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(z2)

    predictions = np.round(A2)
    pre = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("correctness on the training_set: ", str(pre), "%")


if __name__ == '__main__':
    main()
