'''
import cv2 as cv
# 导入CV模块
img = cv.imread('face1.jpg')
# 将图片存入 img 数组中
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#灰度图像的调用
cv.imshow('Gray', gray_img)
#显示灰度图像
cv.imwrite('Gray_face1.jpg', gray_img)
#保存灰度图像 并且将之明名为 Gray_face1，jpg
cv.imshow('read_img', img)
#显示原图片
cv.waitKey(0)
#释放内存
cv.destroyWindow()
'''
