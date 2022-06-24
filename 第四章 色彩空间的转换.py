'''
4.3.2 图像处理实例

Note:
    使用函数 cv2.cvtcolor()
'''
import cv2
import numpy as np

img_ =cv2.imread('Picture_4.jpg')
gray =cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
rgb =cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.imshow('img_', img_)
cv2.imshow('gray', gray)
cv2.imshow('rgb',rgb)

cv2.waitKey()
cv2.destroyAllWindows()

