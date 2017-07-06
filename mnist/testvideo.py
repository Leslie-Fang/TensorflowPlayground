import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
vidcap = cv2.VideoCapture()

vidcap.open(0)
#time.sleep(5)
retval, image = vidcap.retrieve()
plt.imshow(image)
plt.show()
vidcap.release()
cv2.imwrite("test.png", image)
