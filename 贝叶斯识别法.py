'''
Remark:
从总体思路上来说，K近邻算法的确十分先进，
但是数字识别可以说是大材小用了，
主体思路是通过泛值处理获得每个手写体的两个特征向量
并且使用贝叶斯概率学进行计算，从理论上来说是完全可行的
'''
import cv2 as cv
import numpy as np
filename ='digits.png'
t_value =2
img =cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 现在我们将图像分割为5000个单元格，每个单元格为20x20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
img_staff = np.array(cells).astype(np.float32)
# 使其成为一个Numpy数组。它的大小将是（50,100,20,20）
feature_Vector =np.zeros((10,20), dtype=np.float32)
#那么我将feature_Vector 作为训练集
row, col, img_x, img_y= img_staff.shape
for num_index in range(10):
    for staff_index in range(num_index*5,(num_index+1)*5):
        for y in range(col):
            for x in range(img_x):
                feature_Vector[num_index,x] =(np.sum(img_staff[staff_index,y,x])+feature_Vector[num_index,x])/t_value
np.savez('feature_vector.npz',feature_Vector =feature_Vector)
print('Data build successfully! -By HFUT C14')

