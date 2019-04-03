import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('cubes.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

mag,angle = cv.cartToPolar(sobelx,sobely,angleInDegrees=True)


corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
	x,y = i.ravel()	
	#roi = subimg(x,y,gray)
	cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
