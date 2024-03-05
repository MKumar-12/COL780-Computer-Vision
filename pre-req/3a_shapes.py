import cv2
import numpy as np

#Creating a canvas 500x500, 3 channels - for colored img.
canvas = np.zeros((500, 500, 3))                # 3D matrix

#Drawing a line
cv2.line(canvas, (0,0), (100,100), (0,155,0), 2, cv2.LINE_4)            # green-colored line4 from (0,0) to (100,100)
cv2.line(canvas, (0,20), (100,120), (155,0,0), 2, cv2.LINE_8)           # blue-colored line8 from (0,30) to (100,130)
cv2.line(canvas, (0,50), (100,150), (0,0,155), 2, cv2.LINE_AA)          # red-colored lineAA from (0,80) to (100,180)

#Drawing a Rectangle
cv2.rectangle(canvas, (200,200), (250,250), (0,100,250), -1)            # passing -1 as thickness fills the object

#Drawing a Circle
cv2.circle(canvas, (300,350), 20, (200,0,80),3)

#Drawing a Arrowed line
cv2.arrowedLine(canvas, (400,400), (400,500), (255,255,255), tipLength=0.2)


#displaying canvas
cv2.imshow("Canvas", canvas)
cv2.waitKey(10000)