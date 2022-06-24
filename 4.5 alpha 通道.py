'''
在 RGB 色彩空间的三个通道的基础上，还可以再加入
一个 A 通道，称为 alpha 通道，这种 4 通道空间被称为 GRBA 空间
其赋值范围为 [0, 255] 或者 [0,1] 表示从 透明到不透明
'''

'''
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[2, 3, 3], dtype=np.uint8)

bgra = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

print('img =\n',img)
print('bgra=\n',bgra)

# 将四个通道剥离

b,g,r,a = cv2.split(bgra)

print('Alpha=\n', a)

a[:,:] = 125

# 将从 bgra 中提取的 alpha 通道的值设定为 125

# 使用 cv2.merge() 函数 将4个通道 合并
bgra= cv2.merge([b,g,r,a])

print('bgra=\n',bgra)
'''

import cv2
import numpy

img = cv2.imread('Picture_4.jpg')
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
b,g,r,a = cv2.split(bgra)

a[:,:] = 125

bgra_125 = cv2.merge([b,g,r,a])

a[:,:] = 0

bgra_0 = cv2.merge([b,g,r,a])

cv2.imshow('img', img)
cv2.imshow('bgra',bgra)
cv2.imshow('bgra_0',bgra_0)
cv2.imshow('bgra_125',bgra_125)

cv2.imwrite('bgra.png',bgra)
cv2.imwrite('bgra_0.png',bgra_0)
cv2.imwrite('bgra_125.png',bgra_125)


cv2.waitKey()
cv2.destroyAllWindows()





