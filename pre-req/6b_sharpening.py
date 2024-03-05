import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")

#Gaussian Kernel for sharpening
gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)

#Sharpening img using addweighted()
sharpened1 = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)              #overall sum of a & b must be 0     -> applies a.src1 + (b+g).src2
# src1, alpha, src2, beta, gamma(sigmax)
# gamma is used to increase intensity
sharpened2 = cv2.addWeighted(img, 3.5, gaussian_blur, -2.5, 0)
sharpened3 = cv2.addWeighted(img, 7.5, gaussian_blur, -6.5, 0)


#Displaying canvas + sharpened images
cv2.imshow('Original', img)
cv2.imshow('Sharpened 1', sharpened1)
cv2.imshow('Sharpened 2', sharpened2)
cv2.imshow('Sharpened 3', sharpened3)
cv2.waitKey(30000)