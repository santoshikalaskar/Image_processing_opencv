
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5,5),0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    lowerBound=np.array([110,50,50])
    upperBound=np.array([130,255,255])
    mask = cv2.inRange(hsv,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose
    contours, _ = cv2.findContours(maskFinal,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        

    for contour in contours:
        
        area = cv2.contourArea(contour)
        #print(area)
        if area>5000:
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
            cv2.drawContours(frame, contour, -1,(0,255,0),3)
 
    cv2.imshow("Frame",frame)
    cv2.imshow("mask",mask)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
