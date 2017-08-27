import cv2
import numpy as np
 
img=cv2.imread("J.png",0)
#erosion- here we decide the size of a slider, and if all the pixels in the
#area of slider is of same pixels then it moves on else if there is a
#difference then it will remove that different pixel.
#in case of dilation instaed of eroding it pushes the different pixels out
kernel= np.ones((5,5), np.uint8)
erosion=cv2.erode(img, kernel, iterations= 1)
cv2.imshow('erosion', erosion)
dilation=cv2.dilate(img,kernel,iterations=1)
cv2.imshow('dilation', dilation)

 #opening removes false positives(noise in background) and closing removes
#false negatives(noise in the object)
opening= cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('opening',opening)
closing= cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',closing)


cv2.waitKey(0)  
cv2.destroyAllWindows()




