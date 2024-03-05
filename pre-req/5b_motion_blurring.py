import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")
size = 15

#forming filter
kernel = np.zeros((size,size))                    
kernel[int((size-1)/2), :] = np.ones(size)
kernel = kernel / size                              # for normalization : only 1 row with values

#Appling filter to img.
output1 = cv2.filter2D(img, -1, kernel)

#Displaying canvas
cv2.imshow('Original', img)
cv2.imshow('Motion Blur', output1)
cv2.waitKey(30000)