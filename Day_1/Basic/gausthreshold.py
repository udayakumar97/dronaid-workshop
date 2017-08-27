import cv2
import numpy as np
img = cv2.imread('lowlight.jpg')

#Threshold on the color image
retval , threshold = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
cv2.imshow('original',threshold)

#Threshold on grayscale
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold',threshold2)

#Adaptive Threshold
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold',th)



cv2.waitKey(0)
cv2.destroyAllWindows()