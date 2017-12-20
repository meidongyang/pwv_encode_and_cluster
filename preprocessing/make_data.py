#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-13 10:36:00
# @Author  : Mei
# @Link    : https://github.com/meidongyang
# @Version : v0.1
# =============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 数据是做完一阶差分的，没有做标准化
data = pd.read_csv('data.csv', header=0)
data = np.asarray(data)


pwv_ID = data[:, :1].astype(np.str)
raw_data = data[:, 6:].astype(np.float32)

print 'data shape:', data.shape
print 'pwvID shape:', pwv_ID.shape
print 'pwvData shape:', raw_data.shape

waveID = pwv_ID[0:1, :]
pwv_data = raw_data[0:1, :]
print waveID.shape, pwv_data.shape

# 清洗数据，去掉包含很多零值和异常值（非数字、nan等）的数据
for i in xrange(data.shape[0]):
    if (list(raw_data[i, :]).count(0) < 5 and
            list(np.isnan(raw_data[i, :])).count(True) < 1):
        waveID = np.vstack([waveID, pwv_ID[i, :]])
        pwv_data = np.vstack([pwv_data, raw_data[i, :]])

print waveID.shape, pwv_data.shape

# 做标准化
for i in xrange(pwv_data.shape[0]):
    pwv_data[i, :] -= np.mean(pwv_data[i, :], axis=0)
    pwv_data[i, :] /= np.std(pwv_data[i, :], axis=0)

print 'save...'
np.save('data.npy', pwv_data)
np.save('data.npy', waveID)

print pwv_data.shape, waveID.shape


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


# 制作动画
fig, ax = plt.subplots()
pwv_line, = ax.plot(rediff(pwv_data[0, :]))


def updata(data):
    pwv_line.set_ydata(rediff(data))
    return pwv_line


ani = FuncAnimation(fig, updata, pwv_data, interval=100)
print 'show...'
plt.show()
