import cv2 # computer vision library, opencv internally imports NumPy
from ultralytics import YOLO # object detection with pre trained model
import requests # https calls to api
from PIL import Image # image processing library
from io import BytesIO # handles image data in memory
import threading
from dotenv import load_dotenv
import os

model = YOLO('yolov8n.pt') # load a custom yolo model
cap = cv2.VideoCapture(0) # cam object, opens webcam
cv2.namedWindow('Plant Identifier', cv2.WINDOW_NORMAL) # resizable window

plant_cache = {} # dictionary for plant identifications which thread finds
identifying = set() # plants currently being identified

def get_cache_key(bbox): # function to ensure new calls aren't made for every time plant moves a bit
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2 # finds middle
    center_y = (y1 + y2) // 2
    stable_x = (center_x // 50) * 50 # round to nearest 50px
    stable_y = (center_y // 50) * 50
    return f"{stable_x}_{stable_y}"

def crop_plant(frame, bbox): # function to just send plant, not full screen
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2] # takes frame of rows from y1 to y2, and columns from x1 to x2 (array slicing)
    return cropped

def identify_plant(image_crop):
    try:
        rgb_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB) # change bgr to rgb, convertColour(image(array of pixels), convert from bgr to rgb)
        pil_image = Image.fromarray(rgb_crop) # convert numpy array to pil image object

        if max(pil_image.size) > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS) 

        buffer = BytesIO() # in memory file, pil needs to save image to get as bytes so we put it in ram
        pil_image.save(buffer, format='JPEG', quality=90) # save image in buffer (in memory file)
        buffer.seek(0)

        load_dotenv()
        api_key = os.getenv('PLANTNET_API_KEY')

        url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}"
        files = {'images': ('image.jpg', buffer, 'image/jpeg')}

        response = requests.post(url, files=files, timeout=15) # post sends data to server

        if response.status_code == 200:
            results = response.json() # convert json to dictionary
            if 'results' in results and results['results']: # check if results exists in dictionary and not empty
                top = results['results'][0] # get top item for best match
                species = top.get('species', {})
                common_names = species.get('commonNames', [])
                scientific_name = species.get('scientificNameWithoutAuthor', 'Unknown')

                if common_names:
                    name = common_names[0]
                else:
                    name = scientific_name

                score = top.get('score', 0)
                return f"{name} ({score:.2f})"
            else:
                return "No match found"
        else:
            return f"API Error {response.status_code}"
            
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

                if 'plant' in class_name.lower() and confidence > 0.1: # check if plant appears in lowercase class name and confidence is enough
                    bbox_id = get_cache_key((x1, y1, x2, y2))

                    if bbox_id not in plant_cache and bbox_id not in identifying:
                        identifying.add(bbox_id)
                        image_crop = crop_plant(frame, (x1, y1, x2, y2)) # crop function

                        def identify_async(crop, bid): # bid = bbox_id 
                            species = identify_plant(crop) # identify plant function
                            plant_cache[bid] = species
                            identifying.discard(bid)

                        thread = threading.Thread(target=identify_async, args=(image_crop, bbox_id))
                        thread.daemon = True
                        thread.start()

                    if bbox_id in plant_cache:
                        if "No match found" not in plant_cache[bbox_id] and "Error" not in plant_cache[bbox_id]:
                            label = plant_cache[bbox_id]
                            colour = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    elif bbox_id in identifying:
                        label = "Identifying..."
                        colour = (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    cv2.imshow('Plant Identifier', frame) # image show takes in window title and the image to show

    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms for a key press, if press q, exit loop. & 0xFF is so it works better and ord('q') is code for q key
        break

cap.release() # close cam
cv2.destroyAllWindows() # close cv2 windows