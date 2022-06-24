# 在 OpenCV 中，可以使用.bitwise_and() 方法实现按位运算
# 语法格式:.bitwise_and(src1, src2[,mask])
# 任何数值 N and 0 =0
# 任何数值 N(仅考虑八位值) and 255 = N
'''
import cv2
import numpy
a =numpy.random.randint(0,255,(5,5),dtype =numpy.uint8)
b =numpy.zeros((5,5), dtype =numpy.uint8)

b[0:3, 0:3] =255
b[4,4] =255

a_and_b =cv2.bitwise_and(a, b)
print(a)
print(b)
print(a_and_b)
'''

# example 3.8
import cv2
import numpy as np
img_a =cv2.imread('Picture_4.jpg',1)
img_a_p =np.zeros(img_a.shape,dtype=np.uint8)

img_a_p[0:300, 0:200] =255

img_a =cv2.bitwise_and(img_a_p, img_a)
cv2.imshow('New_one',img_a)
cv2.waitKey()
cv2.destroyAllWindows()
'''
# 按位或运算
cv2.bitwise_or()
# 按位非运算
cv2.bitwise_not()
# 按位异或运算
cv2.bitwise_xor()
'''