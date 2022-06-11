# coding=utf-8
import numpy as np
import cv2 as cv

drawing = False #按下鼠标则为真
filepath = 'E:/pythonProject_opencv/手写数字识别/knn_data.npz'

def nothing(x):
    pass

def draw(event,x,y,flags,param):
	global drawing
	if event == cv.EVENT_LBUTTONDOWN:  #响应鼠标按下
		drawing = True
	elif event == cv.EVENT_MOUSEMOVE: #响应鼠标移动
		if drawing == True:
			img[y:y+20,x:x+20] = (255,255,255)
	elif event == cv.EVENT_LBUTTONUP:  #响应鼠标松开
		drawing = False


# 创建一个黑色的图像，一个窗口
img = np.zeros((300,300,3), np.uint8)
cv.namedWindow('image')
# 创建颜色变化的轨迹栏
accuracy = 'accuracy'
clear = 'clear'
distinguish = 'distinguish'
append = 'append'
right = 'right'
data = np.load('knn_data.npz')
train = data['train'].astype(np.float32)
train_labels = data['train_labels'].astype(np.float32)
test = data['test'].astype(np.float32)
test_labels = train_labels.copy()
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

#创建轨迹条
cv.createTrackbar(right,'image',0,9,nothing)   #所写之字的正确数字
cv.createTrackbar(append,'image',0,1,nothing)  #加入训练集中
cv.createTrackbar(distinguish,'image',0,1,nothing)  #识别数字
cv.createTrackbar(clear,'image',0,1,nothing)  #清空画布
cv.createTrackbar(accuracy, 'image',0,1,nothing)  #计算识别率

cv.setMouseCallback('image',draw)
img[:] = (0,0,0) #将画板设为黑色
while(1):
	cv.imshow('image',img)
	if cv.waitKey(1)&0xFF == 27:
		break
	ac = cv.getTrackbarPos(accuracy,'image')
	c = cv.getTrackbarPos(clear,'image')
	d = cv.getTrackbarPos(distinguish,'image')
	a = cv.getTrackbarPos(append,'image')

	#测试正确率
	if ac == 1:
		cv.setTrackbarPos(accuracy, 'image', 0)

		data = np.load(filepath)
		train = data['train'].astype(np.float32)
		train_labels = data['train_labels'].astype(np.float32)
		test = data['test'].astype(np.float32)
		test_labels = train_labels.copy()

		ret, result, neighbours, dist = knn.findNearest(test, k=1)
		matches = result == test_labels
		correct = np.count_nonzero(matches)  # 计算矩阵matches里非零元素个数
		accura = correct * 100.0 / result.size
		print('{:.3f}'.format(accura))

	#清空画布
	if c == 1:
		cv.setTrackbarPos(clear, 'image', 0)
		img[:] = (0,0,0)
	if d == 1: #识别数字
		cv.setTrackbarPos(distinguish,'image',0)
		testimg = img.copy()
		testimg = cv.resize(testimg,(20,20))
		gray = cv.cvtColor(testimg,cv.COLOR_BGR2GRAY)
		x = np.array(gray).reshape(-1, 400).astype(np.float32)
		ret, result, neighbours, dist = knn.findNearest(x, k=5)
		cv.putText(img, str(result[0][0]), (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 200, 200), 1)
	if a == 1:  #加入训练集中
		img[0:50,0:50] = (0,0,0)
		cv.setTrackbarPos(append,'image',0)
		r = cv.getTrackbarPos(right, 'image')
		print('已加入数字 ', str(r))
		r = np.array([[r]]).astype(np.float32)

		testimg = img.copy()
		testimg = cv.resize(testimg, (20, 20))
		gray = cv.cvtColor(testimg, cv.COLOR_BGR2GRAY)
		x = np.array(gray).reshape(-1, 400).astype(np.uint8)
		#将新数据加入训练集中
		train = np.append(train,x,axis=0).astype(np.uint8)
		train_labels = np.append(train_labels,r,axis=0).astype(np.uint8)
		test = np.append(test,x,axis=0).astype(np.uint8)
		#存储进文件
		np.savez(filepath, train=train, train_labels=train_labels, test=test)
cv.destroyAllWindows()
