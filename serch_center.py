import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)

while True:
    _, img = capture.read()
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    conditions = (dst[:,:,0]<150)*1 * ((dst[:,:,0]>90)*1) * (dst[:,:,1]>100)

    sum_item = np.sum(conditions)
    t_conditions = np.transpose(conditions)

    temp = range(len(conditions))
    mean_x = np.sum([temp * t_condition for t_condition in t_conditions])/sum_item
    temp = range(len(t_conditions))
    mean_y = np.sum([temp * condition for condition in conditions])/sum_item

    cv2.circle(img, (mean_y, mean_x),10, (0,0,255), -1)
    cv2.imshow("camera", img)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()