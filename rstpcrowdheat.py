import cv2
import numpy as np
from datetime import datetime, timedelta
import time
from bson.binary import Binary  # Import Binary from bson
#import datetime
from pymongo import MongoClient
import firebase_admin
from firebase_admin import credentials, storage
import base64

# Initialize Firebase Admin SDK with the service account key and storage bucket name
cred = credentials.Certificate("config1.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crowd-91e59.appspot.com'
})


# Access Firebase Storage
bucket = storage.bucket()

net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
client = MongoClient('mongodb+srv://ganeshyadharth:AbleLyf@students.jbrazv2.mongodb.net/?retryWrites=true&w=majority')
mongo_db = client["AbleLyf"]
mongo_collection = mongo_db["crowdentries"]
mongo_collection1 = mongo_db["heatentries"]
cams = mongo_db["camera"]

def getRtsp(camera_name):
    rtsp_link = None
    cursor = cams.find({"camName": camera_name})
    for entry in cursor:
        rtsp_link = entry.get("rtspLink")
        break  # Assuming there's only one entry for the camera name
    return rtsp_link
    
#Change this according to the camera name
camera_name = "demo1"
rtsp_link = getRtsp(camera_name)


distance_thres = 50
crowd_threshold = 3  # Minimum number of persons for crowd detection
save_duration_threshold = 3# Minimum duration of crowd for saving (in seconds)

count = 1
start_time = time.time()
save_interval = 10 # 15 minutes in seconds
elapsed_time = 0

cap = cv2.VideoCapture('data/humans.mp4')
heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

heatWriter = cv2.VideoWriter('output.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))), True)


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
 
            #update heatmap
            heatmap[y:y+h, x:x+w] += 1


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
                output_filename = f'crowd_output_{output_counter}.mp4'
                output_photo = f'crowd_detection_{output_counter}.png'
                output_counter += 1
                writer_output = cv2.VideoWriter(output_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
                crowd_start_frame = 0
                crowd_detected = False
                crowd_timer = 0
                cv2.imwrite(output_photo, img)
                local_file_path = writer_output
                firebase_storage_path = "output/"
                blob = bucket.blob(output_filename)
                blob.upload_from_filename(output_filename)
                download_url = blob.public_url
                signed_url = bucket.blob(output_filename).generate_signed_url(expiration=604800)
                print("Video uploaded to Firebase Storage.")
                #print("Download URL:", download_url)  # Save the image of the crowd
                with open(output_photo, 'rb') as image_file:
                   encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                crowd_entry = {'camName':"6r",'image': f'data:image/jpeg;base64,{encoded_image}', 'crowdDuration': f"{hours} hours {minutes} minutes {seconds} seconds", 'crowdStartTime': crowd_start_time, 'crowdEndTime': crowd_end_time, 'noOfPersons': person, 'detetctedVideo': signed_url }
                #crowdEntry = {'camName': "6r", 'image': encodedImage, 'crowd_duration': f"{hours} hours {minutes} minutes {seconds} seconds", 'crowdStartTime': crowdStartTime, 'crowdEndTime': crowd_end_time, 'noOfPersons': person, 'detetctedvideo': signed_url }

                mongo_collection.insert_one(crowd_entry)
    # Normalize and apply colormap to heatmap
    heatmap_normalized = heatmap / np.max(heatmap)
    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend heatmap with original image
    img_heatmap = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    num = len(persons)
    v = len(groups)  # Number of detected crowds
    #cv2.putText(img_heatmap, 'No of Violations : ' + str(v), (15, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 126, 255), 2)
    cv2.putText(img_heatmap, 'No of persons : ' + str(num), (15, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 126, 255), 2)

   
    heatWriter.write(img_heatmap)

    elapsed_time = time.time() - start_time
    if elapsed_time >= save_interval:
        # Save the graph or perform any other desired action
        heatphoto = f'heatmap_{count}min{num}.png'
        cv2.imwrite(heatphoto, img_heatmap)
        count += 1
        heatDate = datetime.now().strftime('%Y-%m-%d')
        heatTime = datetime.now().strftime('%H:%M:%S')
        start_time = time.time()
        with open(heatphoto, 'rb') as image_file:
            encodedHeat = base64.b64encode(image_file.read()).decode('utf-8')  
        heatEntry = {'camName':"6r",'image': f'data:image/jpeg;base64,{encodedHeat}','date': heatDate, 'time': heatTime, 'noOfPersons': num}
        mongo_collection1.insert_one(heatEntry)

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
