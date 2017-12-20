#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-12 16:23:58
# @Author  : 梅冬阳 (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation

CLUSTER_NUM = 30

encode_data = np.load('encode_data.npy')
spectral = cluster.SpectralClustering(n_clusters=CLUSTER_NUM,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors").fit(encode_data)
core_samples_mask = np.zeros_like(spectral.labels_, dtype=bool)
core_sample_indices_ = xrange(encode_data.shape[0])
core_samples_mask[core_sample_indices_] = True
labels = spectral.labels_

# Number of clusters in labels, ignoring nois if present.
n_clusters_ = len(set(labels))
print 'Estimated number of clusters:', n_clusters_

# Black removed and is used for noise instead.
unique_labels = set(labels)
plt.clf()
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    xy = encode_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=10)

    xy = encode_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=3, alpha=0.5)

plt.title('clusters:%d eigen_solver=arpack affinity=nearest_neighbors' %
          (n_clusters_))
plt.show()

# 输出各个簇中元素个数
for i in xrange(n_clusters_):
    print 'cluster: ', i, '=', list(labels).count(i)


# 数量大于500的类随机打印出100张图并保存到本地
pwvData = np.load('pwv_data.npy')


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


# 路径前面要加 . 否则报错（IOSError: [Errno 13] Permission denied:）
os.makedirs("./cluster_spectral_pic")

for i in set(labels):
    plt.clf()
    pwvData_cluster = pwvData[labels == i]
    count = 0
    if pwvData_cluster.shape[0] >= 20:
        for j in random.sample(xrange(pwvData_cluster.shape[0]), 20):
            # 绘制多子图
            plt.subplot(4, 5, 1 + count)
            plt.plot(rediff(pwvData_cluster[j, :]))
            count += 1
    else:
        for j in xrange(pwvData_cluster.shape[0]):
            # 绘制多子图
            plt.subplot(pwvData_cluster.shape[0], 1, 1 + count)
            plt.plot(rediff(pwvData_cluster[j, :]))
            count += 1

    # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
    plt.savefig("./cluster_spectral_pic/cluster" +
                str(i) + "_" + ".png", dpi=1024)

for i in set(labels):
    pwvData_cluster = pwvData[labels == i]
    # 路径前面要加 . 否则报错（IOSError: [Errno 13] Permission denied:）
    os.makedirs("./Cluster-SpectralClustering_pic/cluster" + str(i))
    if pwvData_cluster.shape[0] > 100:
        random_idx = np.random.randint(0, pwvData_cluster.shape[0], 100)
        for j in random.sample(random_idx, 100):
            plt.clf()
            plt.plot(rediff(pwvData_cluster[j, :]))
            # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
            plt.savefig("./Cluster-SpectralClustering_pic/cluster" +
                        str(i) + "/" + str(j) + ".png")
    else:
        for j in xrange(pwvData_cluster.shape[0]):
            plt.clf()
            plt.plot(rediff(pwvData_cluster[j, :]))
            # 路径前面要加 . 否则报错（IOError: [Errno 2] No such file or directory:）
            plt.savefig("./Cluster-SpectralClustering_pic/cluster" +
                        str(i) + "/" + str(j) + ".png")
print 'save...'
