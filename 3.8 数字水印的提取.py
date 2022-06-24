import cv2
import numpy as np
filepath ='Picture_4_watermask.bmp'
img_ = cv2.imread(filepath, 0)
Height, Width = img_.shape

mask_ = np.ones((Height, Width), dtype= np.uint8)
img_solve = cv2.bitwise_and(img_, mask_)

img_solve_TURE  = img_solve[:, :] >0
img_solve[img_solve_TURE] =255

cv2.imshow('Picture', img_)
cv2.imshow('Solve_picture', img_solve)
cv2.waitKey()
cv2.destroyAllWindows()