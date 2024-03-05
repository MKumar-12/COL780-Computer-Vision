import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")

#forming filter
kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3 = np.ones((3,3), dtype='float32') / 9.0                    #dont forget to normal -> to reduce pixel intensities
kernel_11 = np.ones((11,11), dtype='float32') / 121.0

#Appling filter to img.
output1 = cv2.filter2D(img, -1, kernel_identity)
output2 = cv2.filter2D(img, -1, kernel_3)
output3 = cv2.filter2D(img, -1, kernel_11)

#Displaying canvas
cv2.imshow('Original', output1)
cv2.imshow('Blur_3', output2)
cv2.imshow('Blur_11', output3)
cv2.waitKey(30000)