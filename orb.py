import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cubes.jpg')

orb = cv2.ORB_create()

kp = orb.detect(img,None)

kp,des = orb.compute(img,kp)

img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)
plt.imshow(img2),plt.show()

