import cv2 # computer vision library, opencv internally imports NumPy
from ultralytics import YOLO # object detection with pre trained model
import requests # https calls to api
import base64 # image to text format
from PIL import Image # image processing library
from io import BytesIO # handles image data in memory

model = YOLO('yolov8n.pt') # load a custom yolo model
cap = cv2.VideoCapture(0) # cam object, opens webcam

cv2.namedWindow('Plant Identifier', cv2.WINDOW_NORMAL) # resizable window

def crop_plant(frame, bbox): # function to just send plant, not full screen
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2] # takes frame of rows from y1 to y2, and columns from x1 to x2 (array slicing)
    return cropped

def identify_plant(image_crop):
    try:
        rgb_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB) # change bgr to rgb, convertColour(image(array of pixels), convert from bgr to rgb)
        pil_image = Image.fromarray(rgb_crop) # convert numpy array to pil image object

        if max(pil_image.size) > 512:
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS) 

        buffer = BytesIO() # in memory file, pil needs to save image to get as bytes so we put it in ram
        pil_image.save(buffer, format='JPEG', quality=85) # save image in buffer (in memory file)
        image_bytes = buffer.getvalue() # store buffer data as raw bytes

        image_b64 = base64.b64encode(image_bytes).decode('utf-8') # converts jepg bytes to b64 bytes, then b64 bytes to b64 string (better format)

        url = "https://api.inaturalist.org/v1/computervision/score_image"
        data = {'image': image_b64, 'taxon_id': 47126} # send b64 string (plant photo) and only look for plants

        response = requests.post(url, json=data, timeout=10) # post sends data to server

        if response.status_code == 200:
            results = response.json() # convert json to dictionary
            if 'results' in results and results['results']: # check if results exists in dictionary and not empty
                top = results['results'][0] # get top item for best match
                taxon = top.get('taxon', {})
                name = taxon.get('preferred_common_name') or taxon.get('name', 'Unknown') # get common name, or scientific, or unknown
                conf = top.get('score', 0)
                return f"{name} ({conf:.2f})"
            
            return "Unknown"
            
    except Exception as e:
        return f"Error: {str(e)}"

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
                class_id = int(box.cls[0]) # box.cls (category), [0] is class id, int to convert to whole number. basically turns classification to int?
                class_name = model.names[class_id] # convert numbers to names with id
                confidence = float(box.conf[0]) # confidence score

                if 'plant' in class_name.lower() and confidence > 0.3: # check if plant appears in lowercase class name and confidence is enough
                    image_crop = crop_plant(frame, (x1, y1, x2, y2)) # crop function
                    species = identify_plant(image_crop) # identify plant function
                    label = species
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # cv2 function to draw rectangle on frame, bounded by x1, y1 and x2, y2 (box coords for result object), rgb colour for green, rectangle thickness
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # draw on frame, label for text, others self explanatory

    cv2.imshow('Plant Identifier', frame) # image show takes in window title and the image to show

    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms for a key press, if press q, exit loop. & 0xFF is so it works better and ord('q') is code for q key
        break

cap.release() # close cam
cv2.destroyAllWindows() # close cv2 windows