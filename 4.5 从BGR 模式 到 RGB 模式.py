import cv2
img_ = cv2.imread('Picture_4.jpg')
rgb =  cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

cv2.imshow('img_', img_)
cv2.imshow('rgb',rgb)

cv2.waitKey()
cv2.destroyAllWindows()