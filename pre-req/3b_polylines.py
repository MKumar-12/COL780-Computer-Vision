import cv2
import numpy as np

#Creating a initally black - canvas 400x400, 3 channels - for colored img.
canvas = np.zeros((400, 400, 3))                # 3D matrix

#Required points to be connected
pts = np.array([[10,40], [100,60], [200,300], [170,90]], np.int32)

#Reshape the points to shape : (num_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))


#Drawing a Polygon using pts
cv2.polylines(canvas, [pts], True, (0,200,50), 1)               # T/F indicate whether to join the last & first pt. or not


#displaying canvas
cv2.imshow("Canvas", canvas)
cv2.waitKey(10000)