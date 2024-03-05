import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")

height, width = img.shape[:2]           # height(y),   width(x)

#Translation matrix M
matrix = cv2.getRotationMatrix2D((width/2,height/2), 10, 1)
# center_coord, angle, scale

#Applying translation to img. 
rotated = cv2.warpAffine(img, matrix, (width, height))
# org_img, translation matrix(M), size_of_new_img(dsize)

#Displaying canvas
cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated)
cv2.waitKey(10000)