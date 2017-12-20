#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-09 09:30:46
# @Author  : Mei
# @Link    : https://github.com/meidongyang
# @Version : v0.1

import time
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# load up data
CAE_Data = np.load('encode_data.npy')

print CAE_Data.shape

p = np.array([30, 40, 50, 60, 70, 80, 90, 100])


for i in p:
    # 计时
    tic = time.time()

    tsne_array = TSNE(perplexity=i, n_iter=5000).fit_transform(CAE_Data)
    print 'T-SNE...',
    np.save('TSNEData_5000iter_perplexity' + str(int(i)), tsne_array)
    print i, 'stored...'

    # 计时
    toc = time.time()
    print 'Finish. Run time', int(toc - tic), 's'
