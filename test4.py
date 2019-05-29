import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
width = 400
height = 300
cap.set(3, width)
cap.set(4, width)

_, frame = cap.read()

while(1):
    _, next_frame =  cap.read()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    diff_frame = fgbg.apply(frame)
    diff_frame = fgbg.apply(next_frame)
    #cv2.circle(diff_frame, (width//2, height//2),1, (255,255,255), 1)
    conditions = (diff_frame[:,:]>254)*1
    sum_item = np.sum(conditions)
    if(sum_item != 0):
        t_conditions = np.transpose(conditions)
        temp = range(len(conditions))
        mean_x = np.sum([temp * t_condition for t_condition in t_conditions])//sum_item
        temp = range(len(t_conditions))
        mean_y = np.sum([temp * condition for condition in conditions])//sum_item
        
        cv2.circle(frame, (mean_y, mean_x),15, (255,0,0), 2)
    cv2.imshow('diff_frame',diff_frame)
    cv2.imshow('frame', frame)
    
    frame = next_frame
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()