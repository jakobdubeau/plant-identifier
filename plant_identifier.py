import cv2
from ultralytics import YOLO

model = YOLO('yolo8n.pt') # load a custom yolo model
cap = cv2.VideoCapture(0) # cam object, opens webcam

cv2.namedWindow('Plant Identifier', cv2.WINDOW_NORMAL) # resizable window

while True: # get cam frames forever
    ret, frame = cap.read() # ret = boolean return, frame = image(pixels). So, while capture reads, it returns a true/false if works and the image
    if not ret: # if cam fails exit loop
        break

    frame = cv2.flip(frame, 1) # flip image horizontally, 0 = vertical, 1 = horizontal, -1 = both

    results = model(frame) # takes frame, sends to model, returns a list of object found
    for result in results:
        boxes = result.boxes # boxes = bounding boxes object (3 objects = 3 boxes)
        if boxes is not None:
            for box in boxes: # for each individual box (object)
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # get box coordinates, convert to int. box.xyxy[0] is one detected object with coords, [0] is first set of coords, map int converts coords to int
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # cv2 function to draw rectangle on frame, bounded by x1, y1 and x2, y2 (box coords for result object), rgb colour for green, rectangle thickness

    cv2.imshow('Plant Identifier', frame) # image show takes in window title and the image to show

    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms for a key press, if press q, exit loop. & 0xFF is so it works better and ord('q') is code for q key
        break

cap.release() # close cam
cv2.destroyAllWindows() # close cv2 windows