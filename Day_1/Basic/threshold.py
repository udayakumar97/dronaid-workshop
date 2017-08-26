import cv2
import numpy as np
img1 = cv2.imread('lion.jpg')
cv2.imshow('1',img1)

#Converting image to grayscale
img2gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#Threshold Function
ret, mask = cv2.threshold(img2gray, 225, 255, cv2.THRESH_BINARY)


cv2.imshow('Mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()