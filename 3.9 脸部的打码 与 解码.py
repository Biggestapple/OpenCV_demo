import cv2
import numpy as np
img_ = cv2.imread('Picture_4.jpg' ,0)
Height, Width = img_.shape
mask = np.zeros((Height, Width), dtype= np.uint8)
mask[100:200, 100:300] = 1
# 获取解码 所需的密钥
key =np.random.randint(0,255,(Height, Width), dtype=np.uint8)
# 使用 key  进行加密
img_XorKey = cv2.bitwise_xor(img_, key)
encrypt =cv2.bitwise_and(img_XorKey, mask*255)
# 提取部分的图像
# 将 图像 中部分的值设置为 零
noFace = cv2.bitwise_and(img_, (1-mask)*255)

maskFace = encrypt + noFace
#=========将打码脸解码==========

extractOriginal = cv2.bitwise_xor(maskFace, key)
extract_cutt = cv2.bitwise_and(extractOriginal, mask*255)
extract_cutt_rest = cv2.bitwise_and(maskFace, (1-mask)*255)
extract_img =extract_cutt_rest +extract_cutt

cv2.imshow('img_',img_)
cv2.imshow('key',key)
cv2.imshow('img_XorKey',img_XorKey)
cv2.imshow('encrypt',encrypt)
cv2.imshow('maskFace',maskFace)
cv2.imshow('extract_img',extract_img)
cv2.imshow('extractOriginal',extractOriginal)
cv2.imshow('extract_cutt',extract_cutt)

cv2.waitKey()
cv2.destroyAllWindows()



