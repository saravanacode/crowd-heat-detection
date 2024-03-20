import cv2
import numpy as np
from datetime import datetime
import time
from bson.binary import Binary  # Import Binary from bson

from pymongo import MongoClient

net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
client = MongoClient('mongodb+srv://saravana:samerareddy@cluster1.wgfyfeq.mongodb.net/?retryWrites=true&w=majority')
mongo_db = client["students"]
mongo_collection = mongo_db["crowdentires"]

distance_thres = 50
crowd_threshold = 3  # Minimum number of persons for crowd detection
save_duration_threshold = 3  # Minimum duration of crowd for saving (in seconds)

cap = cv2.VideoCapture('data/humans.mp4')

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
output_counter = 1
writer_output = None

def dist(pt1, pt2):
    try:
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    except:
        return

crowd_start_frame = 0
crowd_detected = False
crowd_timer = 0  # Timer to track the duration of the crowd

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id != 0:
                continue
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    persons = []
    person_centres = []
    violate = set()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            persons.append(boxes[i])
            person_centres.append([x + w // 2, y + h // 2])

    # Create separate groups for crowd detection
    groups = []
    for (x, y, w, h) in persons:
        added = False
        for group in groups:
            if any(dist([x + w // 2, y + h // 2], center) <= distance_thres for center in group):
                group.append([x + w // 2, y + h // 2])
                added = True
                break
        if not added:
            groups.append([[x + w // 2, y + h // 2]])

    for group in groups:
        if len(group) >= crowd_threshold:
            if not crowd_detected:
                crowd_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                crowd_start_time = time.ctime()
                crowd_start_duration = time.time()
                crowd_detected = True
                person = len(group)
            crowd_timer += 1

            x_min = min(person[0] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
            y_min = min(person[1] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
            x_max = max(person[0] + person[2] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
            y_max = max(person[1] + person[3] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)

            # Draw rectangle around the crowd
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(img, 'Crowd Detection', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Check if crowd duration exceeds the threshold
            if crowd_timer >= save_duration_threshold * cap.get(cv2.CAP_PROP_FPS):
                crowd_end_time = time.ctime()
                crowd_end_duration = time.time()
                crowd_duration = crowd_end_duration - crowd_start_duration
                hours = int(crowd_duration // 3600)
                minutes = int((crowd_duration % 3600) // 60)
                seconds = int(crowd_duration % 60)
                print("crowd_duration:", f"{minutes} minutes {seconds} seconds")
                output_filename = f'crowd_output_{output_counter}.avi'
                output_photo = f'crowd_detection_{output_counter}.png'
                output_counter += 1
                writer_output = cv2.VideoWriter(output_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
                crowd_start_frame = 0
                crowd_detected = False
                crowd_timer = 0
                cv2.imwrite(output_photo, img)  # Save the image of the crowd
                with open(output_photo, 'rb') as image_file:
                   encoded_image = Binary(image_file.read())
                crowd_entry = {'image': encoded_image, 'crowd_duration': f"{hours} hours {minutes} minutes {seconds} seconds", 'crowd_start_time': crowd_start_time, 'crowd_end_time': crowd_end_time, 'no of persons': person }
                mongo_collection.insert_one(crowd_entry)

    cv2.imshow("crowd_detections", img)
    if writer_output is not None:
        writer_output.write(img)

    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
if writer_output is not None:
    writer_output.release()
cv2.destroyAllWindows()

