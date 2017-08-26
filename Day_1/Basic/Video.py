import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#CODEC
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#Writing the frames
		out.write(frame)
		
		cv2.imshow('frame',frame)
		cv2.imshow('frame2',gray)


		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
out.release()
cv2.destroyAllWindows()