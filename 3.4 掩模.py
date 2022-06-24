'''
 OpenCV 中很多函数 都会指定一个掩膜
 操作只会在掩膜值非空的像素点上进行，并将其他的像素点的值设置为 零
 如： cv2.add() 函数
'''
# demo 3.10
'''
import cv2
import numpy as np

img_0 =np.ones((4,4), dtype=np.uint8)*3
img_1 =np.ones((4,4), dtype=np.uint8)*3
# 掩码的构造
masks =np.zeros((4,4), dtype=np.uint8)_p
img_ =cv2.add(img_1, img_0,mask=masks)
print('img_0=\n',img_0)
print('img_1=\n',img_1)
print('masks=\n',masks)
print('img_=\n',img_)
'''

# Demo 3.11
import cv2
import numpy as np
img_p =cv2.imread('Picture_1.jpg')
w,h,c =img_p.shape
# 获取长宽 以及数据类型 和 通道数目

masks =np.zeros((w,h),dtype=np.uint8)

masks[100:200,15:260]=1
cv2.imshow('Before',img_p)
img_p =cv2.bitwise_and(img_p, img_p, mask=masks)
cv2.imshow('After',img_p)
print(img_p.shape)
cv2.waitKey()
cv2.destroyAllWindows()




