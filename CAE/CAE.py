#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-13 17:00:00
# @Author  : Mei
# @Link    : https://github.com/meidongyang
# @Version : v3.0
# =============================================

import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pwvData = np.load('pwv_data.npy')
print pwvData.shape

DATA_NUM = pwvData.shape[0]
DATA_LEN = pwvData.shape[1]
BATCH_SIZE = 256
EPOCH = int(np.ceil(DATA_NUM / BATCH_SIZE))
print 'DATA NUM:', DATA_NUM, 'DATA LEN:', DATA_LEN, 'EPOCH:', EPOCH


# 制作 batch
def get_batch_data(dat, idx, size=BATCH_SIZE):
    idx_start = idx * size
    idx_end = idx_start + size

    return dat[idx_start:idx_end, :]


with tf.name_scope('Init'):  # 初始化输入
    # 原始数据为 float64， 所以 placeholder 也为 float64
    raw_data = tf.placeholder(tf.float64, [None, DATA_LEN], name='pwv_data')
    # float64 转换为float32
    cast_data = tf.cast(raw_data, tf.float32, name='cast_type')

    # reshape data
    input_data = tf.reshape(cast_data, shape=[-1, 1, DATA_LEN, 1])

    # Decoder conv2d_T outputs num
    outputs_num = tf.Variable(initial_value=BATCH_SIZE, name='outputs_num')

# Encoder
with tf.name_scope('Encoder'):
    with tf.name_scope('layer_1'):
        en_w_1 = tf.Variable(tf.truncated_normal(
            [1, 4, 1, 8], stddev=0.1, seed=1), name='wights')
        en_b_1 = tf.Variable(tf.zeros([8]), name='bias')
        encode_1 = tf.nn.conv2d(input_data, en_w_1, [
                                1, 1, 2, 1], padding='SAME', name='conv2d')
        relu_1 = tf.nn.relu(tf.add(encode_1, en_b_1), name='ReLU')
        pool_1 = tf.nn.avg_pool(relu_1, ksize=[1, 1, 2, 1], strides=[
                                1, 1, 2, 1], padding='SAME', name='avg_pool')
        print 'layer 1 shape:', pool_1.shape

    with tf.name_scope('layer_2'):
        en_w_2 = tf.Variable(tf.truncated_normal(
            [1, 4, 8, 8], stddev=0.1, seed=1), name='wights')
        en_b_2 = tf.Variable(tf.zeros([8]), name='bias')
        encode_2 = tf.nn.conv2d(
            pool_1, en_w_2, [1, 1, 2, 1], padding='SAME', name='conv2d')
        relu_2 = tf.nn.relu(tf.add(encode_2, en_b_2), name='ReLU')
        pool_2 = tf.nn.avg_pool(relu_2, ksize=[1, 1, 2, 1], strides=[
                                1, 1, 2, 1], padding='SAME', name='avg_pool')
        print 'layer 2 shape:', pool_2.shape

    with tf.name_scope('layer_3'):
        en_w_3 = tf.Variable(tf.truncated_normal(
            [1, 4, 8, 8], stddev=0.1, seed=1), name='wights')
        en_b_3 = tf.Variable(tf.zeros([8]), name='bias')
        encode_3 = tf.nn.conv2d(
            pool_2, en_w_3, [1, 1, 2, 1], padding='SAME', name='conv2d')
        relu_3 = tf.nn.relu(tf.add(encode_3, en_b_3), name='ReLU')
        pool_3 = tf.nn.avg_pool(relu_3, ksize=[1, 1, 2, 1], strides=[
                                1, 1, 2, 1], padding='SAME', name='avg_pool')
        print 'layer 3 shape:', pool_3.shape

# Decoder
with tf.name_scope('Decoder'):
    de_w_1 = tf.Variable(tf.truncated_normal(
        [1, 64, 1, 8], stddev=0.1, seed=1), name='wights')
    decode = tf.nn.conv2d_transpose(pool_3,
                                    de_w_1,
                                    [outputs_num, 1, DATA_LEN, 1],
                                    [1, 1, 64, 1],
                                    name='conv2d_transpose')
    print 'Decode layer shape:', decode.shape
    re_construction = tf.reshape(
        decode, shape=[-1, DATA_LEN], name='decode_data')

# Train
with tf.name_scope('Train'):
    cost = 0.5 * \
        tf.reduce_sum(tf.pow(tf.subtract(re_construction, cast_data), 2))

    # 初始化衰减学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.01, global_step, 30000, 0.98, staircase=False)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost, global_step=global_step)

# 开始会话
with tf.Session() as sess:
    # 初始化
    init = tf.global_variables_initializer()
    # 保存计算图，在 tensorboard 显示
    summary_writer = tf.summary.FileWriter(
        logdir='pwvCAE', graph=tf.get_default_graph())
    summary_writer.close()

    # 运行初始化op
    sess.run(init)

    # 训练 10000 轮
    for i in xrange(200000):
        mean_cost = 0.0
        for j in xrange(EPOCH):
            # 制作 BATCH
            batch_data = get_batch_data(pwvData, j)
            # 开始执行计算图
            _, train_cost = sess.run([optimizer, cost], feed_dict={
                                     raw_data: batch_data})
            # print 'cost:', train_cost
            mean_cost += train_cost
        lr = sess.run([learning_rate])
        print 'epoch', i + 1, 'mean cost=', mean_cost / EPOCH,
        print 'leaning rate:', lr

    # 保存训练好的参数
    save_path = 'CAE_model'

    builder = tf.saved_model.builder.SavedModelBuilder(save_path)
    # inputs, outputs 是 dict，key 对应 value 的 tensor
    inputs = {'input_data':
              tf.saved_model.utils.build_tensor_info(raw_data),
              'outputs_num':
              tf.saved_model.utils.build_tensor_info(outputs_num)}
    outputs = {'encode_data':
               tf.saved_model.utils.build_tensor_info(pool_3),
               're_data':
               tf.saved_model.utils.build_tensor_info(re_construction)}

    # 使用 build_signature_def 方法构建 SignatureDef，第三个参数是 signature 名称
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs,
                                                                       outputs,
                                                                       'CAE')
    builder.add_meta_graph_and_variables(
        sess, ['CAE_model'], {'signature': signature})
    builder.save()
    print '\nModel stored...\n'
