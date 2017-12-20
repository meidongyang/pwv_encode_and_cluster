#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-30 13:09:00
# @Author  : Mei
# @Link    : https://github.com/meidongyang
# @Version : v0.1
# =============================================
'''对数据进行切片，保留一个峰的全部特征
加入正弦波
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

tic = time.time()
pwvData = np.load('data.npy')

# pwvData = pwvData[:, -256:]
pwvData = pwvData[:, 80:190]
print '加入正弦波前：', pwvData.shape
# 随机生成2998条正弦波数据 shape: [2998, 110]
for i in xrange(2998):
    sinValue = 20 * np.sin(np.arange(0, 2 * np.pi,
                                     0.0575) + np.random.randn()) + 15
    pwvData = np.vstack((pwvData, sinValue))
print '加入正弦波后：', pwvData.shape


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


# 创建画布
fig, ax = plt.subplots()
# 输入第一条 Y 轴数据
pwv = rediff(pwvData[0, :])
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
ani = FuncAnimation(fig, update, pwvData, interval=100)
plt.ylim(-8, 40)
print 'Ploting...'
plt.show()

print 'Done...'
print 'Saving...'
np.save('data.npy', pwvData)
print 'Done...'
toc = time.time()
print toc - tic, 's'
