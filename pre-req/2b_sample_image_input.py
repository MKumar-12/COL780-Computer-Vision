import cv2 

# channel order is BGR in color mode
# image = cv2.imread('img1.png', cv2.IMREAD_COLOR)                  # filename, channel mode - colour{default}
# image = cv2.imread('img1.png', cv2.IMREAD_UNCHANGED)              # filename, channel mode - when using on CT-scans, X-rays
image = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)              # filename, channel mode


#to display it onto screen
cv2.imshow('Test img', image)
cv2.waitKey(5000)                               # by-default waits for any i/p key stroke to exit, otherwise pass values in ms