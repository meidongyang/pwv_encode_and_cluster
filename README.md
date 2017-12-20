# 脉搏波数据聚类分析
## 预处理
1 对数据进行清洗，去掉包含很多零值和异常值（非数字、nan等）的数据
2 对数据做一阶差分、标准化
3 找出波形的所有拐点，检验该点是否为上升沿斜率最大点，然后对相邻的两个上升沿斜率最大点进行切片，得到一个心跳周期的波形数据，对一个心跳周期的波形数据进行重采样拉伸操作，拉伸为统一长度

## 使用卷积自编码对数据进行压缩处理
![CAE_model](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/CAE.png)

$$图1 自编码网络模型$$

filter、步长情况如下：

| layer | conv2d | avg_pool |
| --- | --- | --- |
| 1 |<div>filter[1,4,1,8]</div> <div>strides[1, 1, 2, 1]</div>|<div>ksize[1, 1, 2, 1]</div> <div>strides[1, 1, 2, 1]</div> |
| 2、3 |<div>filter[1,4,8,8]</div> <div>strides[1, 1, 2, 1]</div>|<div>ksize[1, 1, 2, 1]</div> <div>strides[1, 1, 2, 1]</div>|

经过20万轮迭代后，长度256位的数据可压缩至24位

![原始数据vs重构数据](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/encode.png)

$$图2 原始数据(蓝)与重构数据(红)可视化误差对比$$

![encodeData](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/encode_data.png)

$$图3 经过自编码压缩后的数据可视化$$

![encodeData_reshape_2dim](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/dim2.png)

$$图4 将压缩数据转换为图片形式$$

## 降维
因sklearn包里的聚类API还会对较长数据进行PCA降维，所以使用t-SNE降维生成多组数据进行对比

## DBSCAN
![DBSCAN](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/WX20171220-092607.png)

## 谱聚类
![谱聚类](https://github.com/meidongyang/pwv_encode_and_cluster/blob/master/pic/WX20171220-092557.png)
