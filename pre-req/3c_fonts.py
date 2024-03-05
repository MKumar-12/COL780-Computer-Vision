import cv2
import numpy as np

#Creating a initally black - canvas 400x400, 1 channels - for B&W img.
canvas = np.zeros((800, 500))                # 2D matrix

#List of fonts
fonts = [cv2.FONT_HERSHEY_PLAIN,
         cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_COMPLEX_SMALL,
         cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
         cv2.FONT_ITALIC]


pos = (10,30)

for i in range(0, 9):
    # (canvas_obj, txt, coord, font_family, scale, color, thickness, line_format)
    cv2.putText(canvas, "OpenCV!", pos, fonts[i], 1, (255,255,255), 1, cv2.LINE_AA)                 
    pos = (pos[0], pos[1] + 30)

    cv2.putText(canvas, "OpenCV!".lower(), pos, fonts[i], 1, (255,255,255), 1, cv2.LINE_AA)
    pos = (pos[0], pos[1] + 30)


#displaying canvas
cv2.imshow("Canvas", canvas)
cv2.waitKey(10000)