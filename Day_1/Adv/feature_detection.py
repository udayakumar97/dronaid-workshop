#in feature matching we dont need the pic to be found in the given pic to have same
# lighting, angle, rotation etc. like template matching
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1= cv2.imread('opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)


orb= cv2.ORB_create()
#we will use orb as a detector to detect features

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)
#finding out key points and their descriptions with orb detector

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#this is BFMatcher object.

matches=bf.match(des1,des2)
matches=sorted(matches,key=lambda x:x.distance)
#Here we create matches of the descriptors, then we sort them based on their distances
img3= cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
#10 denotes the no.of matches to be shown
#as we inc no.of matches limit then false positives  may also be matched
plt.imshow(img3)
plt.show()


