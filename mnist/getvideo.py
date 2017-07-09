import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

cap = cv2.VideoCapture(0)
n = 0
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    #cv2.imshow("capture", frame)
    plt.imshow(frame,cmap = 'gray', interpolation = 'bicubic')
    plt.show(block=False)
    plt.pause(3)
    #time.sleep(3)
    plt.close()
    print n
    n = n + 1
    if n > 1:
        break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
cap.release()
#cv2.destroyAllWindows()