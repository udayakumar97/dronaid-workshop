import cv2
import numpy as np
import matplotlib
cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()    


    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)

    lower_red = np.array([30,150,50])
    upper_red= np.array([255,255,180])

    #converts the range given to white and rest is black
    #cv2.inRange(image,lower range pixel,upper range pixel)
    mask = cv2.inRange(hsv, lower_red, upper_red) 
    cv2.imshow('mask', mask)



    #erosion- here we decide the size of a slider, and if all the pixels in the
    #area of slider is of same pixels then it moves on else if there is a
    #difference then it will remove that different pixel.
    #in case of dilation instaed of eroding it pushes the different pixels out
    kernel= np.ones((5,5), np.uint8)
    erosion=cv2.erode(mask, kernel, iterations= 1)
    cv2.imshow('erosion', erosion)
    dilation=cv2.dilate(mask,kernel,iterations=1)
    cv2.imshow('dilation', dilation)

    #opening removes false positives(noise in background) and closing removes
    #false negatives(noise in the object)
    opening= cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    cv2.imshow('opening',   opening)
    closing= cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    cv2.imshow('closing',closing)


    k= cv2.waitKey(5) & 0xFF 
    if k==27: 
        break #esc key to stop

cv2.destroyAllWindows()
cap.release()



