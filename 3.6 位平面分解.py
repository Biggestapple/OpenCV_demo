'''
在八位灰度图中， 数值范围为 0~255 为8 位二进制
二进制的不同位 剥离 为 a0 a2 ... a7
其中 a0 为最低位，权重最低 对图像的影响最小
其中 a7 为最高位，权重最大 对图像的影响最大

在八位灰度图中，可以将原图分解为八个位平面
针对 RGB 图像 可以将 RGB 三通道分解 ，再将 三通道 逐一分解合并
称为 原始图像的 N 位分解
'''

'''
图像预处理具体步骤：
构造提取矩阵：
元素值均为 2 的 N 次方
提取位平面：
将灰度图像与提取矩阵进行 按位与运算
10110001 预进行 第四位分解
10110001 --and 00010000 ---> 00010000
10100001 --and 00010000 ---> 00000000
'''

'''
note：
提取位平面 也可以通过将 二进制像素值 右移至 指定位 ，然后对二 取模得到
'''

'''
4.范值处理

将 普通数 转化 为 255 即 位图纯色
语句：
mask =RD[:, :, i] >0
mask 将 RD 中大于 零 的元素提取出来
大于零的元素 --->TRUE
等于小于零的元素 --->FALSE

RD[mask] =255

对 mask 对应为 TRUE 的值 赋值

'''
# Demo 3.5.1
import  cv2
import numpy as np

img_ =cv2.imread('Picture_3.jpg')
cv2.imshow('Before', img_)
img_b =img_[:, :, 1]
cv2.imshow('Get_bit_map',img_b)
wi,hi =img_b.shape
masks =np.zeros((wi, hi, 8),dtype=np.uint8)
# 相当于 八个滤镜 作用
#初始化提取矩阵
for index in range(0, 8):
    masks[ :, :, index] =2**index
for index in range(0, 8):
    img_trans =cv2.bitwise_and(masks[ :, :, index], img_b)
    # 范值处理
    mask =img_trans >0
    img_trans[mask] =255
    cv2.imshow(f'RD-{index}',img_trans)
cv2.waitKey()
cv2.destroyAllWindows()






















