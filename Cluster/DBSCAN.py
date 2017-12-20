#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-12 16:23:58
# @Author  : 梅冬阳 (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

# import matplotlib
# matplotlib.use('Agg')
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.animation import FuncAnimation

# Load data
encode_data = np.load("TSNEData_5000iter_perplexity30.npy")
print encode_data.shape


# Compute pearson
def cal_pearson(v1, v2):
    '''皮尔逊相关度的计算结果在两者完全匹配的情况下为1.0，而在两者毫无关系的情况下则为0.0。

        return 1.0 - num / den
    目的是为了让相似度越大的两个元素之间的距离变得更小。
    '''
    v1_ = v1 - np.mean(v1)
    v2_ = v2 - np.mean(v2)
    return np.dot(v1_, v2_) / (np.linalg.norm(v1_) * np.linalg.norm(v2_))


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


eps = 0.18
min_samples = 8

# Compute DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(encode_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_
print 'Estimated number of labels:', len(set(labels))
# Number of clusters in labels, ignoring nois if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print 'Estimated number of clusters:', n_clusters_

# Black removed and is used for noise instead.
unique_labels = set(labels)
plt.clf()
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = encode_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=10)

    xy = encode_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=3, alpha=0.5)

plt.title('clusters:%d eps:%.4f samples:%d' % (n_clusters_, eps, min_samples))
plt.show()

# 输出各个簇中元素个数
for i in xrange(n_clusters_ + 1):
    print 'cluster: ', i - 1, '=', list(labels).count(i - 1)

pwvData = np.load('pwv_data.npy')


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


# 每类大于100的打印100张图，小于100的全打印
for i in set(labels):
    pwvData_cluster = pwvData[labels == i]
    # 路径前面要加 . 否则报错（IOSError: [Errno 13] Permission denied:）
    os.makedirs("./cluster_DBSCAN_pic/cluster" + str(i))
    if pwvData_cluster.shape[0] > 100:
        random_idx = np.random.randint(0, pwvData_cluster.shape[0], 100)
        for j in random.sample(random_idx, 100):
            plt.clf()
            plt.plot(rediff(pwvData_cluster[j, :]))
            # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
            plt.savefig("./cluster_DBSCAN_pic/cluster" +
                        str(i) + "/" + str(j) + ".png")
    else:
        for j in xrange(pwvData_cluster.shape[0]):
            plt.clf()
            plt.plot(rediff(pwvData_cluster[j, :]))
            # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
            plt.savefig("./cluster_DBSCAN_pic/cluster" +
                        str(i) + "/" + str(j) + ".png")

# 打印四个波形组一个图
# 路径前面要加 . 否则报错（IOSError: [Errno 13] Permission denied:）
os.makedirs("./dbscan")

for i in set(labels):
    plt.clf()
    pwvData_cluster = pwvData[labels == i]
    if pwvData_cluster.shape[0] >= 4:
        count = 0
        for j in random.sample(xrange(pwvData_cluster.shape[0]), 4):
            # 绘制多子图
            plt.subplot(221 + count)
            plt.plot(rediff(pwvData_cluster[j, :]))
            count += 1

    # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
    plt.savefig("./dbscan/cluster" + str(i) + "_" +
                str(list(labels).count(i)) + ".png")


print 'save...'


for i in range(n_clusters_):
    if num[i] > 500:
        pwvData_cluster = pwvData[labels == idx[i]]
        random_idx = np.random.randint(0, pwvData_cluster.shape[0], 100)
        # 创建画布
        fig, ax = plt.subplots()
        # 输入第一条 Y 轴数据
        pwv = rediff(pwvData_cluster[0, :])
        # 初始化动画的第一条曲线
        pwvLine, = ax.plot(pwv)
        # 曲线更新函数
        # 如果是参数是list,则默认每次取list中的一个元素,即list[0],list[1],...

        def update(data):
            # 调用 set_ydata() 更新 Y 轴数据
            pwvLine.set_ydata(rediff(data))
            return pwvLine
        # 创建 FuncAnimation 对象定时调用 update()
        # interval 参数为每秒的帧数
        ani = FuncAnimation(fig, update, pwvData_cluster, interval=100)
        plt.ylim(-8, 40)
        print 'Ploting...'
        plt.show()
