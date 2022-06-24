'''
将需要阴藏的信息 嵌入载体图像的 最低有效位
即最低图层 ，从而达到隐藏信息的效果


原理：
嵌入过程：
将载体的第零位替换为 数字水印信息
提取过程：
将载体图像的最低有效位 所构成的第 0 个位面 提取出来，从而得到数字水印 信息



实现方法：

读取原始图像，并获取原始图像的 行数和 列数
建立提取矩阵
保留载体图像的高七位， 将最低位置零
111101001
  and
111111110

OUTPUT:

111101000

note:
要保留图像的高七位 ，还可以先将像素右移一位的位操作
再将像素左移一位
水印图像的处理

先将灰度二值图像 W
进行泛值处理
将 255 值 转换为 1

嵌入水印：
使用 或运算 的方法

最后显示图像：
流程图：

1,初始化:

2,载体图像处理

3,建立提取矩阵

4,载体图像的最低位置零

5,水印图像处理

6,嵌入水印

7,显示图像

8,结束


'''
# Demo 3.8.3
import  cv2
import  numpy as np

img_ = cv2.imread('Picture_4.jpg', 0)

watermark = cv2.imread('water_mark.bmp', 0)
# 将图像的通道剥离
# 接下来是 泛值处理
# 将水印图像内的值 255 处理为 1， 以方便嵌入
w_m = watermark[:, :] >0
watermark[w_m] = 1
# 读取原始图像的 大小值 即 shape 值

Height,Width = img_.shape
watermark_cut = watermark[:Height, :Width]
#===============嵌入的过程=================

t_254 = np.ones((Height, Width), dtype=np.uint8)*254

# 生成提取数组

img_H7 = cv2.bitwise_and(img_, t_254)

img_w = cv2.bitwise_or(img_H7,watermark_cut )

cv2.imshow('After', img_w)
cv2.imshow('Befor', img_)
cv2.imwrite('Picture_4_watermask.bmp',img_w)

cv2.waitKey()
cv2.destroyAllWindows()

































