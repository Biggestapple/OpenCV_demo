'''
包含三个基本要素
H:色调
S:饱和度
V:亮度

Key:
通过 inrange 函数锁定特定值
Opencv 中通过 cv2.inrange() 来判断图像内像素点的像素
值是否在指定的范围内

语法格式：

dst = cv2.inRange(src, lower, upper)

note:
1,如果 src 的值处于指定的区间内部，则 dst 中对应的值为 255

2,如果 src 的值不处于指定的区间内部，则 dst 中对应的值为 0
'''

# demo 4.7
import cv2
import numpy
lowerp,upperp = 10, 100
img_ = numpy.random.randint(0, 256,size=[5, 5], dtype=numpy.uint8)

img_range = cv2.inRange(img_,lowerp, upperp )

print('Img_inRange : \n',img_range)
print('Img_ : \n',img_)

# 返回 结果 Img_inRange 可以理解为 掩码数组






