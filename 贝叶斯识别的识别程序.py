'''
这里是算法的核心部分
'''
import numpy as np
import cv2 as cv

filename = 'feature_vector.npz'
threaHold = [
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.2,
]
erro = 0
all_ex = 0
# 首先我们先导入数据
data = np.load(filename)
feature_Vector = data['feature_Vector']
num, col = feature_Vector.shape
err_bar = np.zeros((10, 20), dtype=np.float32)
print('The shape of Feature_Vector is:', feature_Vector.shape)


def distinguish_num(num_img):
    for index in range(num):
        for i in range(col):
            err_bar[index, i] = np.abs(np.sum(num_img[i]) * threaHold[i] - feature_Vector[index, i])


def give_num():
    ans = []
    for i in range(10):
        ans.append(np.sum(err_bar[i]))
    return ans


def num_output():
    return give_num().index(min(give_num()))


img = cv.imread('digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 现在我们将图像分割为5000个单元格，每个单元格为20x20,进行训练
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
img_staff = np.array(cells).astype(np.float32)


# 接下来开始训练矫正
def num_correct(img, index):
    for i in range(col):
        if np.sum(img[i]) > feature_Vector[index, i]:
            threaHold[i] = 1 - feature_Vector[index, i] / np.sum(img[i]) * 0.2
        elif np.sum(img[i]) < feature_Vector[index, i]:
            threaHold[i] = 1 + np.sum(img[i]) * 0.4 / feature_Vector[index, i]
        else:
            pass
row, col_, img_x, img_y = img_staff.shape
for num_index in range(10):
    for staff_index in range(num_index * 5, (num_index + 1) * 5):
        for i in range(col_):
            distinguish_num(img_staff[staff_index, i])
            if num_output() != num_index:
                num_correct(img_staff[staff_index, i], num_index)
print('Generate successfully!! -HFUT:', threaHold)


def accurate_calu():
    global all_ex, erro
    for num_index in range(10):
        for staff_index in range(num_index * 5, (num_index + 1) * 5):
            for i in range(col_):
                distinguish_num(img_staff[staff_index, i])
                # print(num_output())
                all_ex = all_ex + 1
                if num_output() != num_index:
                    erro = erro + 1
    print(f'The accurate is {(all_ex - erro) / all_ex * 100} %')
    return f'{(all_ex - erro) / all_ex * 100} %'


'''
创建用户交互界面
'''
drawing = False  # 按下鼠标则为真


def nothing(x):
    pass


def draw(event, x, y, flags, param):
    global drawing
    if event == cv.EVENT_LBUTTONDOWN:  # 响应鼠标按下
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:  # 响应鼠标移动
        if drawing == True:
            img_window[y:y + 20, x:x + 20] = (255, 255, 255)
    elif event == cv.EVENT_LBUTTONUP:  # 响应鼠标松开
        drawing = False


img_window = np.zeros((300, 300, 3), np.uint8)
cv.namedWindow('image')
# 创建颜色变化的轨迹栏
accuracy = 'accuracy'
clear = 'clear'
distinguish = 'dist-num'
append = 'append'
right = 'right'
cv.createTrackbar(right, 'image', 0, 9, nothing)  # 所写之字的正确数字
cv.createTrackbar(append, 'image', 0, 1, nothing)  # 加入训练集中
cv.createTrackbar(distinguish, 'image', 0, 1, nothing)  # 识别数字
cv.createTrackbar(clear, 'image', 0, 1, nothing)  # 清空画布
cv.createTrackbar(accuracy, 'image', 0, 1, nothing)  # 计算识别率
cv.setMouseCallback('image', draw)
img_window[:] = (0, 0, 0)  # 将画板设为黑色
flag =0
while (1):

    cv.imshow('image', img_window)
    if cv.waitKey(1) & 0xFF == 27:
        break
    ac = cv.getTrackbarPos(accuracy, 'image')
    c = cv.getTrackbarPos(clear, 'image')
    d = cv.getTrackbarPos(distinguish, 'image')
    a = cv.getTrackbarPos(append, 'image')
    if ac == 1:
        cv.setTrackbarPos(accuracy, 'image', 1)
        cv.putText(img_window, accurate_calu(), (5,290), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 200, 200), 1)
        cv.setTrackbarPos(accuracy,'image', 0)
    if c == 1:
        cv.setTrackbarPos(clear, 'image', 1)
        img_window[:] = (0, 0, 0)
        flag =0
    if d == 1:  # 识别数字
        cv.setTrackbarPos(distinguish, 'image', 1)
        target_img =img_window.copy()
        target_img= cv.resize(target_img, (20, 20))
        target_img=cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
        distinguish_num(target_img)
        cv.putText(img_window, f'{num_output()}', (5, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 200, 200), 1)
    if a == 1 and flag ==0:
        #重新特征定位
        flag =1
        cv.setTrackbarPos(append, 'image', 1)
        r = cv.getTrackbarPos(right, 'image')
        print('Adding numbers --HFUT ', str(r))
        if r!=num_output():
            try:
                num_correct(target_img, r)
                cv.putText(img_window, f'Adding successfully --HFUT', (5, 30), cv.FONT_HERSHEY_PLAIN, 1, (100, 200, 200), 1)
            except NameError:
                cv.putText(img_window, f'Please dist-num please', (5, 30), cv.FONT_HERSHEY_PLAIN, 1,
                           (100, 200, 200), 1)
            else:
                pass
cv.destroyAllWindows()
