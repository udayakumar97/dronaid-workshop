import numpy as np
import cv2

img = cv2.imread('scene.jpg',cv2.IMREAD_COLOR)

#		(var_name, co ordinates,color,thickness)
cv2.line(img,(0,0),(200,300),(255,255,255),50)

cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)

#			 (var_name, center,radius ,color, fill=-1)
cv2.circle(img,(447,63), 63, (0,255,0), -1)


pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)

#			 ((var_name, co ordinates, close shape=True , color, thickness))	
cv2.polylines(img, [pts], True, (0,255,255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
#   ( img_var,'text',Position, font, f_size, (color), thickness)
cv2.putText(img,'Open_CV',(10,200), font, 3, (100,15,200), 10)

cv2.imshow('image',img)


cv2.waitKey(0)
cv2.destroyAllWindows()