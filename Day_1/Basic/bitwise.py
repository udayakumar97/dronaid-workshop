import cv2
import numpy as np

# Load two images
img1 = cv2.imread('scene.jpg')
img2 = cv2.imread('lion.jpg')

#Taking dimensions of img2
rows,cols,channels = img2.shape

#creating ROI(Region of Image) of img1 using dimensions from img2
roi = img1[0:rows, 0:cols ]

cv2.imshow('Image 1',img1)


img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# add a threshold
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY)
cv2.imshow('Mask',mask)

mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
cv2.imshow('img1_bg',img1_bg)


# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
cv2.imshow('img2_fg',img1_bg)


dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()