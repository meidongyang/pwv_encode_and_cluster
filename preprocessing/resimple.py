# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.animation import FuncAnimation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 1


# 重采样，拉伸波形数据。原始数据的采样率为122Hz
def resample(ts, origin_sampling_rate,
             new_sampling_rate,
             order=3):
    '''
    resample a time series
    ts: input time series
    origin_sampling_rate: sampling rate of the input time series
    new_sampling_rate: sampling rate of output time series
    order: Degree of the smoothing spline
    '''
    # modeling a time series as a cubic spline function
    data_size = ts.shape[0]
    static_size = 256
    half_QuJian = int(static_size/2)
    time_length = float(data_size-1)/origin_sampling_rate
    idx, step = np.linspace(0.0,
                            time_length,
                            num=data_size,
                            endpoint=True,
                            retstep=True)
    f = InterpolatedUnivariateSpline(idx, ts, k=order)

    # resampling the time series
    resampling_ps = int(time_length*new_sampling_rate)
    # the tail may be lost, so the time length should be re-calculate
    time_length_cuttail = float(resampling_ps-1)/new_sampling_rate
    newx, new_step = np.linspace(0.0,
                                 time_length_cuttail,
                                 num=resampling_ps,
                                 endpoint=True,
                                 retstep=True)
    newy = f(newx)
    len_newx = newx.shape[0]
    middle_index = int(len_newx/2)
    # final = np.zeros((1,256))
    final_data = newy[
        middle_index-half_QuJian:middle_index+static_size-half_QuJian]
    # print(final_data.shape)
    final_data_array = np.zeros((1, static_size))

    i = 0
    for k in final_data:
        final_data_array[0, i] = k
        i += 1
    # print('original____________',final_data_array.shape)
    return final_data_array


# 制作batch
def get_sub_matrix(dat, idx, size=BATCH_SIZE):
    rstart = idx * size
    rend = rstart + size

    return dat[rstart:rend, :]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x4(x, pooling_factor):
    pooling_factor *= 4
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME'),\
        pooling_factor


def max_pool_1x2(x, pooling_factor):
    pooling_factor *= 2
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME'),\
        pooling_factor


def avg_pool_1x2(x, pooling_factor):
    pooling_factor *= 2
    return tf.nn.avg_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME'),\
        pooling_factor


class CNN_simple:
    conv_len = 8
    num_conv1 = 16
    num_conv2 = 32
    num_fullc = 256
    pooling_factor = 1
    useGPU = False

    sess = None
    x = None
    W_conv1 = None
    b_conv1 = None
    h_conv1 = None
    h_pool1 = None
    W_conv2 = None
    b_conv2 = None
    h_conv2 = None
    h_pool2 = None
    W_fc1 = None
    b_fc1 = None
    h_pool2_flat = None
    h_fc1 = None
    keep_prob = None
    h_fc1_drop = None
    W_fc2 = None
    b_fc2 = None
    y_conv = None

    def __init__(self,
                 feature_num,
                 label_dim,
                 my_path_to_model,
                 useGPU=False,
                 scope_name='models_simple'):
        with tf.variable_scope(scope_name):
            print("CNN param:")
            print("conv_len %d, num_conv1 %d, num_conv2 %d, num_fullc %d" % (
                self.conv_len, self.num_conv1, self.num_conv2, self.num_fullc))
            self.useGPU = False
            print('If useGPU:' + str(self.useGPU))
            # choose CPU or GPU
            if self.useGPU:
                self.sess = tf.InteractiveSession()
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
                self.sess = tf.InteractiveSession(config=config)
            # Create the model
            pooling_factor = 1
            # input data
            self.x = tf.placeholder(tf.float32, [None, feature_num])
            input_x = tf.reshape(self.x, [-1, 1, feature_num, 1])

            # conv2d - layer1
            self.W_conv1 = weight_variable(
                [1, self.conv_len, 1, self.num_conv1])
            self.b_conv1 = bias_variable([self.num_conv1])
            self.h_conv1 = tf.nn.relu(
                conv2d(input_x, self.W_conv1) + self.b_conv1)

            # pooling - layer1
            self.h_pool1, pooling_factor = max_pool_1x4(
                self.h_conv1, pooling_factor)

            # conv2d - layer2
            self.W_conv2 = weight_variable(
                [1, self.conv_len, self.num_conv1, self.num_conv2])
            self.b_conv2 = bias_variable([self.num_conv2])
            self.h_conv2 = tf.nn.relu(
                conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

            # pooling - layer2
            self.h_pool2, pooling_factor = max_pool_1x2(
                self.h_conv2, pooling_factor)

            # full connected weights dimension
            pooled_size = int(feature_num / pooling_factor * self.num_conv2)

            print("pooling:(%d, %d) -> %d" %
                  (feature_num, pooling_factor, pooled_size))

            # full connected - layer1
            self.W_fc1 = weight_variable([pooled_size, self.num_fullc])
            self.b_fc1 = bias_variable([self.num_fullc])
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, pooled_size])
            print(self.h_pool2_flat.shape)

            self.h_fc1 = tf.nn.relu(
                tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            print(self.h_fc1.shape)

            # dropout
            self.keep_prob = tf.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
            print(self.h_fc1_drop.shape)

            # full connected - layer2
            self.W_fc2 = weight_variable([self.num_fullc, label_dim])
            self.b_fc2 = weight_variable([label_dim])
            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
            print('y_conv.shape=', self.y_conv.shape)
            self.y_conv_sotfmax = tf.nn.softmax(self.y_conv)

            # 模型的路径
            self.my_path_to_model = my_path_to_model
            # 为 get_model_checkpoint() 恢复模型函数先声明一个 Saver
            self.saver = tf.train.Saver()

    # 输出 inference，one-hot 格式
    def predict(self, x):
        y_predict = self.sess.run(self.y_conv_sotfmax,
                                  feed_dict={
                                      self.x: x,
                                      self.keep_prob: 1.0})
        return y_predict

    # 恢复模型
    def get_model_checkpoint(self):
        # self.saver.restore(self.sess, self.my_path_to_model)
        ckpt = tf.train.get_checkpoint_state(self.my_path_to_model)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model loaded from %s" % (ckpt.model_checkpoint_path))
            return True
        else:
            print("No checkpoint file found in " + self.my_path_to_model)
            return False


pwvData = np.load('data.npy')
pwvID = np.load('data_ID.npy')
print('pwvData shape =', pwvData.shape)
print('pwvID shape=', pwvID.shape)

simple = CNN_simple(feature_num=256,
                    label_dim=4,
                    my_path_to_model='models')

if simple.get_model_checkpoint():
    print("model restored...")
else:
    print("error...")

'''
1 数据长度为400，求拐点不够长，所以需要对最前、最后各128位数据做反转镜像，然后进行拼接
2 以拐点为中心放入CNN模型找出斜率上升最大点（输出的数组第一个值最大则为斜率上升最大点）
3 找出两个斜率上升最大点的数据，第一条中间点的后半段+第二条中间点前半段是一个心跳周期的波形
4 每个人的心率不同，要对截取的一个心跳周期的波形进行拉伸——重采样
搞定
'''
resample_data = []
data_ID = []

for i in xrange(pwvData.shape[0]):
    if i % 1000 == 0:
        print('No.', i)

    data = pwvData[i:i + 1, :]

    # 对数据前、后 128 位数据做反转、拼接
    re_data_1 = list(data[0, :128])
    re_data_1.reverse()
    re_data_2 = list(data[0, -128:])
    re_data_2.reverse()
    data = re_data_1 + list(data[0]) + re_data_2

    data = np.asarray(data).reshape((1, len(data)))
    # print(data.shape)

    min_set = []
    max_set = []
    min_idx = np.argmax(data[0])
    max_idx = np.argmin(data[0])

    # 求拐点
    for j in np.arange(200, 500):
        # 求极小值的下标
        if (data[0][j + 1] >= data[0][j] and data[0][j - 1] >= data[0][j]):
            min_idx = j
            min_set.append(min_idx)
            min_idx = np.argmax(data[0])

        # 求极大值的下标
        if (data[0][j + 1] <= data[0][j] and data[0][j - 1] <= data[0][j]):
            max_idx = j
            max_set.append(max_idx)
            max_idx = np.argmin(data[0])

    # print('min set:', min_set, len(min_set))
    # print('max set:', max_set, len(max_set))

    min_set = np.asarray(min_set)
    max_set = np.asarray(max_set)
    idx_set = np.hstack((min_set, max_set))
    idx_set.sort()

    pwv_simple = []
    pwv_simple_ID = []

    # 截取的下标
    idx1 = 0
    idx2 = 0
    count = 0
    for j in idx_set:
        pick = simple.predict(data[:, j - 128:j + 128])
        # print('pick=', pick)
        if pick[0][0] > 0.5:
            count += 1
            if count == 1:
                idx1 = j
                # print('idx 1 =', idx1)
            if count == 2 and (j - idx1) > 10:
                idx2 = j
                # print(idx1, idx2)
                break

    # 判断是否找到斜率上升最大点
    if idx1 == 0 or idx2 == 0:
        continue

    idx1 -= 128
    idx2 -= 128
    # print('idx1=', idx1, 'idx2=', idx2),

    # 计算重采样的采样率
    new_s_r = 256 / ((idx2 - idx1) / 122) + 6
    # print('sample rate:', new_s_r)
    # 重采样-数据拉伸
    re = np.asarray(resample(pwvData[i, idx1:idx2], 122, new_s_r))
    resample_data.append(re[0])
    data_ID.append(pwvID[i, :])

resample_data = np.asarray(resample_data)
data_ID = np.asarray(data_ID)
print('finally data shape:', resample_data.shape)
print('wave ID shape:', data_ID.shape)


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


# 可视化查看数据(制作动画)
fig, ax = plt.subplots()
pwv_line, = ax.plot(rediff(resample_data[0, :]))


def updata(data):
    pwv_line.set_ydata(rediff(data))
    return pwv_line


ani = FuncAnimation(fig, updata, resample_data, interval=100)
print('show...')
plt.show()
# 将动画保存为mp4格式视频(需要ffmpeg)
# ani.save('Visualization.mp4')
# print('\nVideo stored...')

re_pwv_data = resample_data[0:1, :]
re_pwv_data_id = data_ID[0:1, :]
# 再次清洗数据，去掉包含很多零值和异常值（非数字、nan等）的数据
for i in xrange(1, resample_data.shape[0]):
    if (list(resample_data[i, :]).count(0) < 50 and
            list(np.isnan(resample_data[i, :])).count(True) < 1):
        re_pwv_data = np.vstack([re_pwv_data, resample_data[i, :]])
        re_pwv_data_id = np.vstack([re_pwv_data_id, data_ID[i, :]])
    else:
        print('No.%d data error...' % i)

# 没问题，保存数据
np.save('data', re_pwv_data)
np.save('data_id', re_pwv_data_id)
print('save...')
