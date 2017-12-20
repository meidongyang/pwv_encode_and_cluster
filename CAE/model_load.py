#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-21 14:16:39
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

signature_key = 'signature'
input_key = 'input_data'
outputs_num = 'outputs_num'
encode_key = 'encode_data'
output_key = 're_data'

pwvData = np.load('pwv_data.npy')

# 模型路径
path = 'CAE_model_pb'


def rediff(data):
    '''一阶差分的逆向操作
    '''
    re_data = np.zeros(data.shape[0], dtype=float)
    re_data[0] = data[0]
    for k in range(1, data.shape[0]):
        re_data[k] = re_data[k - 1] + data[k]
    return re_data


with tf.Session() as sess:
    # 加载模型
    load_meta = tf.saved_model.loader.load(sess, ['CAE_model'], path)
    # 取出 SignatureDef 对象
    signature = load_meta.signature_def
    # 从 signature 中根据 tensor name 找出具体的 tensor
    input_data = signature[signature_key].inputs[input_key].name
    data_num = signature[signature_key].inputs[outputs_num].name
    output_data = signature[signature_key].outputs[output_key].name

    # 获取 tensor 进行 inference
    input_data = sess.graph.get_tensor_by_name(input_data)
    data_num = sess.graph.get_tensor_by_name(data_num)
    output_data = sess.graph.get_tensor_by_name(output_data)

    # 输出重构的数据
    re = sess.run(output_data, feed_dict={input_data: pwvData,
                                          data_num: pwvData.shape[0]})
    print re.shape

    # 制作动画 - 重构数据与原始数据可视化对比
    y1 = pwvData[0, :]
    y2 = re[0, :]

    fig, ax = plt.subplots()

    # 创建动态可视化对象，设置 animated 参数为 True
    line1, = ax.plot(rediff(y1), animated=True,
                     label='$RawInput$', color='b')
    line2, = ax.plot(rediff(y2), animated=True,
                     label='$Reconstruction$', color='r', alpha=0.6)

    def update_line(i):
        y1 = pwvData[i, :]
        y2 = re[i, :]
        # 调用set_ydata()更新Y轴数据
        line1.set_ydata(rediff(y1))
        line2.set_ydata(rediff(y2))
        # 在动画中回调函数 update_line() 中设置所有动画元素的数据
        # 参数 i 为当前的显示帧数，这里使用帧数修改波形，并返回一个包含所有动画元素的序列
        return line1, line2

    # 创建 FuncAnimation 对象定时调用 update_line()
    # interval 参数为每秒的帧数
    # blit 为 True 表示使用缓存加速每帧图像的绘制
    # frames 参数设置最大帧数
    # update_line() 的帧数参数将在 0～epoch-1 之前循环变化
    ani = FuncAnimation(fig, update_line, blit=True,
                        interval=100, frames=pwvData.shape[0])
    # 设置Y轴的显示范围
    plt.ylim(-80, 50)
    # 显示图示（即标签 RawInputs 和 Reconstruction）
    plt.legend()
    print 'show...'
    plt.show()
    # 将动画保存为mp4格式视频(需要ffmpeg)
    ani.save('Reconstruction_vs_raw_150.mp4')
    print '\nVideo stored...'

    # 从 signature 中根据 tensor name 找出具体的 tensor
    encode = signature[signature_key].outputs[encode_key].name
    # 获取 tensor 进行 inference
    encode = sess.graph.get_tensor_by_name(encode)
    # 输出 Encoder 后的数据
    encode_data = sess.run(encode, feed_dict={input_data: pwvData,
                                              data_num: pwvData.shape[0]})
    print encode_data.shape
    dim1 = encode_data.shape[0]
    dim2 = encode_data.shape[1] * encode_data.shape[2] * encode_data.shape[3]
    encode_data = encode_data.reshape(dim1, dim2)
    print encode_data.shape

    # 保存到本地
    np.save('encode_data', encode_data)
