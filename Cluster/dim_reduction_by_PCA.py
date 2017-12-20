#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-13 15:24:57
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import time


# Load data
data = np.load('CAE_DATA.npy')

DATA_NUM = data.shape[0]
DATA_LEN = data.shape[1] * data.shape[2]

# Reshape data
pcaInput = np.reshape(data, (DATA_NUM, DATA_LEN))

# PCA
pca = PCA(n_components=0.98).fit(pcaInput)
pcaData = pca.transform(pcaInput)

# explained_variance_ratio_: 降维后的各主成分的方差值占总方差值的比例。比例越大，则越是重要的主成分。
print pca.explained_variance_ratio_
# explained_variance_: 降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
print pca.explained_variance_
print pca.n_components_
print pcaData.shape

# 计时开始
tic = time.time()


# Compute pearson
def cal_pearson(v1, v2):
    '''皮尔逊相关度的计算结果在两者完全匹配的情况下为1.0，而在两者毫无关系的情况下则为0.0。

        return 1.0 - num / den
    目的是为了让相似度越大的两个元素之间的距离变得更小。
    '''
    # 简单求和
    sum1 = np.sum(v1)
    sum2 = np.sum(v2)

    # 求平方和
    sum1Sq = np.sum([v ** 2 for v in v1])
    sum2Sq = np.sum([v ** 2 for v in v2])

    # 求乘积之和
    pSum = np.sum([v1[i] * v2[i] for i in xrange(len(v1))])

    # 计算 r (Pearson score)
    num = pSum - (sum1 * sum2 / len(v1))
    den1 = sum1Sq - (sum1 ** 2) / len(v1)
    den2 = sum2Sq - (sum2 ** 2) / len(v1)
    den = np.sqrt(den1 * den2)
    if den == 0:
        return 0

    return 1.0 - num / den


# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=5, metric=cal_pearson).fit(pcaData)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# 计时结束
toc = time.time()

# Number of clusters in labels, ignoring nois if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print 'Estimated number of clusters:', n_clusters_

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = pcaData[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=6)

    xy = pcaData[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(
        col), markeredgecolor='k', markersize=3)

plt.title('Estimated number of clusters:%d' % n_clusters_)
plt.savefig("30000_pcaclusters.png")
plt.show()
