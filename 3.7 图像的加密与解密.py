'''
使用异或运算实现：
xor(a,b)=c
xor(c,b)=a
实现解密的过程
'''
import cv2
import numpy as np

img_a=cv2.imread('Picture_4.jpg', 0)
height,width = img_a.shape
key = np.random.randint(0,256,size=[height, width], dtype=np.uint8)
encryption = cv2.bitwise_xor(img_a, key)
decryption = cv2.bitwise_xor(encryption, key)

cv2.imshow('Before', img_a)
cv2.imshow('After', encryption)
cv2.imshow('Then', decryption)

cv2.waitKey()
cv2.destroyAllWindows()