import numpy as np
import cv2

image = np.zeros((500,500))                      #initialize 500x500 grid with all pixels = black    (Grayscale value -> 0)
# image[:,:] = 100            
image = image[:,:] + 20                          #lighten all pixels {causing Brightness}

image[100:300, 250:350] = 255                    #set specific pixels as White   (Grayscale value -> 0)

cv2.imwrite('sample.png', image)                #save the img. matrix to an .png file