import cv2
import numpy as np
from matplotlib import pyplot as plt

def subimg(x,y,img):
	a = x-8
	b = y-8
	c = 0
	d = 0
	if a<0:
		c = -a
		a = 0
	if b<0:
		d = -b
		b = 0
	c = x+7+c
	d = y+7+d
	return img[a:c,b:d]

#def findCorners (mag,angle,thresh):


img = cv2.imread('cubes.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,50,0.01,10)
corners = np.int0(corners)

for i in corners:
	x,y = i.ravel()
	roi = subimg(x,y,gray)	
	sx = cv2.Sobel(roi,cv2.CV_64F,1,0,ksize=5)
	sy = cv2.Sobel(roi,cv2.CV_64F,0,1,ksize=5)
	mag,angle = cv2.cartToPolar(sx,sy,angleInDegrees=True)
	#matches = findCorners(mag,angle,roi)
	cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()


