import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('cubes.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

#image gradients
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

mag,angle = cv.cartToPolar(sobelx,sobely,angleInDegrees=True)


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.show()


plt.subplot(2,2,1),plt.imshow(mag,cmap = 'gray')
plt.title('Magnitude'),plt.xticks([]),plt.yticks([])

plt.show()

#cannny edge detection
edges = cv.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


