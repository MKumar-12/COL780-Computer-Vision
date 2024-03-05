import cv2

# Create instance of video capture
cap = cv2.VideoCapture(0)                       # 0 -> take webcam feed, even mpeg video, or an Ip-cam feed can be passed as a parameter
opened = cap.isOpened()                         # chks if cam opened successfully or not

# fourcc - 4 char-code {for saving video-feed} to save CODEC
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Fetch meta-data from Feed :
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)

print(height)
print("FPS : {}".format(fps))

#initiate video writer
out = cv2.VideoWriter('temp.avi', fourcc, fps, (int(width), int(height)), True)

if(opened) :
    while(cap.isOpened()) :
        ret, frame = cap.read()                         #read every frame, & chk if valid using ret boolean
        if(ret) :
            cv2.imshow("Cam feed", frame)
            out.write(frame)                            #save frame to video_dump
            if(cv2.waitKey(2) == 27) :                  #exit if the user presses ESC
                break   

out.release()
cap.release()
cv2.destroyAllWindows()