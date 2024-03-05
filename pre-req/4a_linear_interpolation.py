import numpy as np
import cv2

#reading a sample img
img = cv2.imread('img1.png')

#img resizing to specific amt. of pixels    -> org_canvas, new_dimensions
# img_resized = cv2.resize(img, (500,500))


#Img. resizing using Linear Interpolation
img_re_linear = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)       
#canvas, new_dim (OPTIONAL), factor_x_multiplier, factor_y_multiplier, interpolation_type

#Img. resizing using Cubic Interpolation
img_re_cubic = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)       


#displaying all 3 imgs :
cv2.imshow('Original', img)
cv2.imshow('Linear', img_re_linear)
cv2.imshow('Cubic', img_re_cubic)

if(cv2.waitKey() == 27) :           # close all windows on ESC
    cv2.destroyAllWindows()