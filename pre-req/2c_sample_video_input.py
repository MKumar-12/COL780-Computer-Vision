import cv2

# Create instance of video capture
cap = cv2.VideoCapture(0)                       # 0 -> take webcam feed, even mpeg video, or an Ip-cam feed can be passed as a parameter
opened = cap.isOpened()                         # chks if cam opened successfully or not

# Fetch meta-data from Feed :
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)

print(height)
print("FPS : {}".format(fps))


if(opened) :
    while(cap.isOpened()) :
        ret, frame = cap.read()                         #read every frame, & chk if valid using ret boolean
        if(ret) :
            cv2.imshow("Cam feed", frame)
            if(cv2.waitKey(2) == 27) :                  #exit if the user presses ESC
                break   