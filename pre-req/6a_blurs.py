import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")
rows, cols = img.shape[:2]

#Kernel blurring
kernel_5 = np.ones((5,5), dtype='float32') / 25.0                    #dont forget to normal -> to reduce pixel intensities
output_kernel = cv2.filter2D(img, -1, kernel_5)

#Blur fn.
output_blur = cv2.blur(img, (5,5))                                  
# img, kernel_size (as tuple x,y)

#box filter
output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)         #normalize = true bydefault                                  

#Gaussian blurring
output_gaus = cv2.GaussianBlur(img, (3,3), 0)
# img, kernel_size, sigmax (std_dev in x-dir) 

#Median blur
output_med = cv2.medianBlur(img, 5)

#Bilateral blur
output_bil = cv2.bilateralFilter(img, 10, 6, 6)
# img, diameter_to_consider, convolves_k_nearest_colors (into 1), sigmaspace


#Displaying canvas
cv2.imshow('Original', img)
cv2.imshow('Blur_5', output_kernel)
cv2.imshow('Blur() output', output_blur)
cv2.imshow('Box filter', output_box)
cv2.imshow('Gaussian', output_gaus)
cv2.imshow('Median blur', output_med)
cv2.imshow('Bilateral', output_bil)
cv2.waitKey(60000)