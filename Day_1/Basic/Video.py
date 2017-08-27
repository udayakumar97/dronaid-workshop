import cv2
import numpy as np 

#Starts Video Capture
cap = cv2.VideoCapture(0)

#CODEC - Compression-Decompression of raw video file 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 #fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while True:
		#Start Reading frames from cap
		ret, frame = cap.read()
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#Writing the frames
		out.write(frame)
		
		cv2.imshow('frame',frame)
		cv2.imshow('frame2',gray)

		#Quit loop if q is entered
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

#Releasing thread so that it can be used again
cap.release()
out.release()
cv2.destroyAllWindows()
