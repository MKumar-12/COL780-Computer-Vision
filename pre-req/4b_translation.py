import numpy as np
import cv2

#reading a sample img
img = cv2.imread("img1.png")

#Translation matrix M
trans_matrix = np.float32([[1, 0, 100], [0, 1, 100]])                  # offset img by (100,100)

#Applying translation to img. 
translated = cv2.warpAffine(img, trans_matrix, (img.shape[1] + 100, img.shape[0] + 100))
# org_img, translation matrix(M), size_of_new_img(dsize)

#Displaying canvas
cv2.imshow('Original', img)
cv2.imshow('Translated', translated)
cv2.waitKey(10000)