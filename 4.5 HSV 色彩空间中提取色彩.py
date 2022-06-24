'''
note:
在提取颜色的过程中，往往不是提取一个特定的值 ，而是提取一个颜色区间
在 HSV 模式内 :

*RED:   [0,100,100] -- [10,255,255]
*BLUE:  [110,100,100] -- [130,255,255]
*GREEN: [50,100,100] -- [70,255,255]

'''

import cv2
import numpy as np

img_ = cv2.imread('Picture_1.jpg')
HSV = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

cv2.imshow('img_', img_)
#=======指定蓝色值的范围==========
minBLUE =np.array([110,100,100])
maxBLUE =np.array( [130,255,255])

# 创建 更加高效的 数组方法

# 确定蓝色的区域
# 通过 HSV 通道
img_mask = cv2.inRange(HSV, minBLUE, maxBLUE)
img_blue = cv2.bitwise_and(img_, img_, mask= img_mask )
cv2.imshow('img_blue',img_blue)
#=======指定绿色值的范围==========
minGREEN = np.array([50,100,100])
maxGREEN = np.array([70,255,255])
# 确定绿色的区域
# 通过 HSV 通道
img_mask = cv2.inRange(HSV, minGREEN, maxGREEN)
img_green = cv2.bitwise_and(img_, img_, mask= img_mask )
cv2.imshow('img_green',img_green)
#=======指定红色值的范围==========
minRED = np.array([0,100,100])
maxRED = np.array([10,255,255])
# 确定红色的区域
# 通过 HSV 通道
img_mask = cv2.inRange(HSV, minRED, maxRED)
img_red = cv2.bitwise_and(img_, img_, mask= img_mask )
cv2.imshow('img_red',img_red)

cv2.waitKey()
cv2.destroyAllWindows()