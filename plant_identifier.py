import cv2

cap = cv2.VideoCapture(0) # cam object, opens webcam

cv2.namedWindow('Plant Identifier', cv2.WINDOW_NORMAL) # resizable window

while True: # get cam frames forever
    ret, frame = cap.read() # ret = boolean return, frame = image(pixels). So, while capture reads, it returns a true/false if works and the image
    if not ret: # if cam fails exit loop
        break

    frame = cv2.flip(frame, 1) # flip image horizontally, 0 = vertical, 1 = horizontal, -1 = both

    cv2.imshow('Plant Identifier', frame) # image show takes in window title and the image to show

    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms for a key press, if press q, exit loop. & 0xFF is so it works better and ord('q') is code for q key
        break

cap.release() # close cam
cv2.destroyAllWindows() # close cv2 windows