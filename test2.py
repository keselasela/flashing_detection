import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
cap.set(3, 400)
cap.set(4, 300)

_, frame = cap.read()

while(1):
    
    _, next_frame =  cap.read()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    diff_frame = fgbg.apply(frame)
    diff_frame = fgbg.apply(next_frame)
    cv2.imshow('frame',diff_frame)
    print(diff_frame)
    frame = next_frame
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()