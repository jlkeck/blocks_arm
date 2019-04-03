import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cubes.jpg')

fast = cv2.FastFeatureDetector_create()

kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

cv2.imwrite('fast_true.png',img2)

fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

img3 = cv2.drawKeypoints(img,kp, None,color=(255,0,0))

cv2.imwrite('fast_false.png',img3)

plt.imshow(img3),plt.show()
