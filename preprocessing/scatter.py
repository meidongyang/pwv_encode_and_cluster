#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-24 14:45:58
# @Author  : 梅冬阳 (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import xlrd
import numpy as np
import matplotlib.pyplot as plt

# 读取excel表
data = xlrd.open_workbook('data.xlsx')

# 获取第3张工作表， table.row_values(i) 行，table.col_values(i) 列
table = data.sheets()[2]

# 读取pwv、欧姆龙的值
pwv = np.asarray(table.col_values(5))
omron_L = np.asarray(table.col_values(6))

# 转为float64
pwv = pwv[1:].astype('float64')
omron_L = omron_L[1:].astype('float64')

# 1阶拟合（线性拟合）
c = np.polyfit(pwv[1:], omron_L[1:], deg=1)
fitX = np.linspace(750, 2300)  # 拟合曲线的起止长度
fitY = np.polyval(c, fitX)

# 画图
plt.figure()
# 画散点图
plt.scatter(pwv[1:], omron_L[1:])
# 画拟合线
plt.plot(fitX, fitY, label='$Fitted Curve$', c='r')
plt.xlabel('Volcano')
plt.ylabel('Omron')
plt.title(u'Critical  Value')
plt.legend()
plt.show()
