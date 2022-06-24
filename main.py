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


'''
import cv2
a = cv2.imread('face1.jpg')
b =a
result1=b+a
result2=cv2.add(a, b)
cv2.imshow('a', a)
cv2.imshow('result1', result1)
cv2.imshow('result2', result2)
cv2.waitKey()
cv2.destroyWindow()
'''

'''
import cv2
result = cv2.imread('face1.jpg',)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyWindow()
'''

#2.2 像素处理
'''
import cv2
import numpy as np # 将numpy 的名字重定义为 np ，使书写更加方便

img = np.zeros((8,8),dtype=np.uint8) # 生成 8*8 数组 类型为 unit8

print("Img= \n",img)

cv2.imshow("one",img)

cv2.waitKey()
cv2.destroyWindow()
'''
'''
import  cv2
import numpy as np
img = cv2.imread("Picture_1.jpg")
cv2.imshow("Before_img",img)

for i in range(1,50):
    for j in range(1,50):
        img[i][j]=0
cv2.imshow("Now_img",img)
cv2.waitKey()
cv2.destroyWindow()
'''


# 对于单通道的实验 三维数组的应用
'''
import  numpy as np
import cv2
blue = np.zeros((300,300,3),dtype=np.uint8)
blue[:,:,0]=255
print("Blue:",blue)
cv2.imshow("BLUE",blue)

green = np.zeros((300,300,3),dtype=np.uint8)
green[:,:,1]=255
print("Green:",green)
cv2.imshow("Green",green)

red = np.zeros((300,300,3),dtype=np.uint8)
red[:,:,2]=255
print("red:",red)
cv2.imshow("Red",red)
cv2.waitKey()
cv2.destroyWindow()
'''
'''
import cv2
import numpy as np
img = np.zeros((300,400,3),dtype=np.uint8)
img[:,100:250,0]=255
img[:,200:350,1]=255
img[:,330:390,2]=255
# 创造彩色图片
# 促使颜色相交
cv2.imshow("IMG_c",img)
# 具备彩色的条纹

cv2.waitKey()
cv2.destroyWindow()
'''

# 2.6 运用 numpy,array 访问像素 语句 item() itemset()
'''
import cv2
import numpy as np
img =np.random.randint(10,99,size=[256,256],dtype=np.uint8) # 一维 灰度图 范围为 10到 99 之间
cv2.imshow("Demo",img)

cv2.waitKey()
cv2.destroyWindow()
'''
'''
import  cv2
import numpy
import cv2
import numpy as np
img =np.random.randint(0,255,size=[256,256,3],dtype=np.uint8) # 三维 彩色随机图 范围为 256到 0 之间
cv2.imshow("Demo_Before",img)

#img[20:30,20:30,1]=0;
#img[20:30,20:30,2]=0;

img[0:255,0:255,2]=0;
img[0:255,0:255,0]=0; 

#img[50:70,90:110,0]=0;
#img[50:70,90:110,1]=0;

#也可以使用循环 进行逐一赋值 的操作
#循环的嵌套

#数组的改变值 操作
cv2.imshow("Demo_After",img)

cv2.waitKey()
cv2.destroyWindow()
'''

'''
import  cv2
import numpy
import cv2
import numpy as np
img =np.random.randint(0,255,size=[256,256,3],dtype=np.uint8)
cv2.imwrite("Test_demo.jpg",img)
'''

# Numpy 的运用重要

# 二维数组的生成
# 使用 item()及 itemset() 进行读取与修改

'''
import  numpy as np
img = np.random.randint(10,99,size=[5,5],dtype=np.uint)
# 生成完毕

print("img =\n",img)

print("Read [3,4]",img.item(2,3))
# 根据 退一 原则 （2,3）实际上对应的是三行 ，第四列 的元素
# 就像 C语言中的那样

# 更加 高效地修改值

img.itemset((2,3),255)

print("Read [3,4]",img.item(2,3))
'''

'''
# 模拟三维数组
import  numpy as np
img = np.random.randint(10,99,size=[5,5,3],dtype=np.uint) # 三通道 对应 rgb
# 生成完毕

print("img =\n",img)

print("Read [3,4,2]",img.item(2,3,1))
# 根据 退一 原则 （2,3）实际上对应的是三行 ，第四列 的元素,第 2 维度 元素
# 就像 C语言中的那样

# 更加 高效地修改值

img.itemset((2,3,1),255)

print("Read [3,4,2]",img.item(2,3,1))
'''

# 生成彩色图像
'''
import  cv2
import numpy as np
img =np.random.randint(0,255,size=[255,255,3],dtype=np.uint8) # unint8 必须添加
# 生成完毕

cv2.imshow("Random_pic",img)
cv2.waitKey()
cv2.destroyWindow()
'''


'''
import  cv2
import numpy as np
img =np.random.randint(0, 255, size=[256, 256, 3], dtype=np.uint8)
# unint8 必须添加
# 生成完毕

cv2.imshow("Random_pic_before", img)
print("Read[1,2,1]", img.item((0, 1, 0)))
print("Read[25,200,2]", img.item((24, 199, 1)))
print("Read[255,2,1]", img.item((245, 1, 0)))

# 修改 img 部分

for y_p in range(0, 64):
    for x_p in range(0, 64):
        for colo in range(0, 3):
            img.itemset((y_p, x_p, colo), 255)

cv2.imshow("Random_pic_After", img)
cv2.waitKey()
cv2.destroyWindow()
'''

# PART 2.4  by python 3.7
# ROI  区域

#Demo 2.13
'''
import cv2

img = cv2.imread('Picture_1.jpg')

# 将图片剥离

img_n = img[0:200,50:200]

#X坐标 值为50像素至150 像素中的值

cv2.imshow('Before',img)
cv2.imshow('After',img_n)

cv2.waitKey()
cv2.destroyWindow()
'''

'''
import numpy as np
import cv2

img_f = cv2.imread('Picture_2.jpg')
face_cover = np.random.randint(0,256,(500,500,3))
# 生成随机码图
cv2.imshow('Before',img_f)

img_f[400:900,400:900]=face_cover
# 覆盖原图像

cv2.imshow('Before',img_f)
cv2.waitKey()
cv2.destroyWindow()
'''

'''
import cv2
import numpy as np

roi = cv2.imread('Picture_2.jpg')
ROI = roi[400:732,400:668]

img = cv2.imread('Picture_1.jpg')

img = img+ROI

cv2.imshow('Exchange',img)
cv2.imwrite('Exchange.jpg',img)

cv2.waitKey()
cv2.destroyWindow()
'''
# RGB 通道操作
# By python 3,7
'''
import cv2
lena =cv2.imread('Picture_2.jpg')

img_b =lena[: ,: ,0]
img_g =lena[: ,: ,1]
img_r =lena[: ,: ,2]

cv2.imshow('img_b',img_b)
cv2.imshow('img_r',img_r)
cv2.imshow('img_g',img_g)



lena[: ,: ,0]=0
cv2.imshow('No blue part',lena)

cv2.waitKey()
cv2.destroyWindow()

'''

# 通过函数拆分图像通道
# cv2.split() 函数的运用
'''
import cv2

img = cv2.imread('Picture_2.jpg')

b =cv2.split(img)[0]
g =cv2.split(img)[1]
r =cv2.split(img)[2]

cv2.imshow('img_b',b)
cv2.imshow('img_r',g)
cv2.imshow('img_g',r)

cv2.waitKey()
cv2.destroyWindow()
'''

# 2.5.2 RGB 通道的合并
# 使用 函数cv2.merge([b ,g ,r])
# cv2.merge([b ,g ,r]) 是混合正常图像的顺序

'''
import cv2

img = cv2.imread('Picture_2.jpg')

b =cv2.split(img)[0]
g =cv2.split(img)[1]
r =cv2.split(img)[2]

cv2.imshow('img_b',b)
cv2.imshow('img_r',g)
cv2.imshow('img_g',r)

rgb =cv2.merge([r, g, b])
bgr =cv2.merge([b, g, r])

cv2.imshow('rgb',rgb)
cv2.imshow('bgr',bgr)


cv2.waitKey()
cv2.destroyWindow()
'''

# 2.6 获取图像属性
# 使用 .shape .size .dtype 函数

# demo 2.6
'''
import cv2
img =cv2.imread('Picture_2.jpg')


print('Img.shape',img.shape)
print('Img.size',img.size)
print('Img.dtype',img.dtype)
'''

# 第三章 图像运算

# 3.1 图像的加法运算
'''
import numpy as np
import cv2

img_1 =np.random.randint(0,255,[3,3],dtype=np.uint8)
img_2 =np.random.randint(0,255,[3,3],dtype=np.uint8)

print('img_1\n',img_1)
print('img_2\n',img_2)
print('img_1+img_2\n',img_1+img_2)
'''

# 运算法则： (A+B)%256 = A+B

# 3.1.2 cv2.add()函数的运用
# 函数的运算法则
#  if (A+B)<=255 --->cv2.add( A + B )= A+B;
#  if (A+B)>255 --->cv2.add( A + B )= 255;
'''
import numpy as np
import cv2

img_1 =np.random.randint(0,255,[3,3],dtype=np.uint8)
img_2 =np.random.randint(0,255,[3,3],dtype=np.uint8)

print('img_1\n',img_1)
print('img_2\n',img_2)
print('img_1+img_2\n',cv2.add(img_1,img_2))
'''

'''
import cv2
import numpy as np

img =cv2.imread('Picture_2.jpg')
cv2.imshow('Result_',img+img)
cv2.imshow('Result_add',cv2.add(img,img))

cv2.waitKey()
cv2.destroyWindow()
'''

'''
import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5,minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#       print(f'Face in {(x+w)/2},{(y+h)/2}')
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
'''

# 3.2 图像的加权和
# 使用函数 cv2.addWeighted(src1, alpha, src2, beta, gamma)
# 公式 : src1*alpha + src2*beta + gamma = OUTPUT
# 第三位变量一般作为亮度条件
# demo 3.4


'''
import numpy as np
import cv2
'''


''''
img_1 =np.ones([3,4],dtype=np.uint8)*100
img_2 =np.ones([3,4],dtype=np.uint8)*10

img =cv2.addWeighted(img_1,0.5,img_2,1,5)

print(img)
'''

'''
roi = cv2.imread('Picture_2.jpg')
ROI = roi[400:732,400:668]

img = cv2.imread('Picture_1.jpg')

img = cv2.addWeighted(ROI,0.6,img,0.4,5)

cv2.imshow('Exchange',img)
cv2.imwrite('Exchange_2.jpg',img)

cv2.waitKey()
cv2.destroyWindow()
'''

# demo 3.1.2 something intresting
'''
roi = cv2.imread('Picture_2.jpg')
ROI = roi[400:732,400:668]
img = cv2.imread('Picture_1.jpg')

for value in range(0, 10):
    img = cv2.addWeighted(ROI, value/10, img, (10-value)/10, 5)
    cv2.imshow('Exchange', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
'''















