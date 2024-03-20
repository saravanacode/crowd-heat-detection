import cv2
import numpy as np
import cvzone

net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

distance_thres = 50
crowd_threshold = 3  # Minimum number of persons for crowd detection
save_duration_threshold = 2  # Minimum duration of crowd for saving (in seconds)

cap = cv2.VideoCapture('data/humans.mp4')

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))), True)

heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

def dist(pt1, pt2):
    try:
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    except:
        return

ret = True
crowd_start_frame = 0
crowd_detected = False
output_counter = 1

while ret:
    ret, img = cap.read()
    if ret:
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

                # Update heatmap
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
                    crowd_detected = True
                    crowd_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                x_min = min(person[0] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
                y_min = min(person[1] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
                x_max = max(person[0] + person[2] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)
                y_max = max(person[1] + person[3] for person in persons if [person[0] + person[2] // 2, person[1] + person[3] // 2] in group)

                # Draw rectangle around the crowd
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(img, 'Crowd Detection', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if crowd_detected:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame >= crowd_start_frame + save_duration_threshold * cap.get(cv2.CAP_PROP_FPS):
                output_filename = f'output_{output_counter}.avi'
                output_counter += 1
                writer_output = cv2.VideoWriter(output_filename, fourcc, 30, (int(cap.get(3)), int(cap.get(4))), True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, crowd_start_frame)
                while cap.get(cv2.CAP_PROP_POS_FRAMES) < current_frame:
                    ret, frame = cap.read()
                    if ret:
                        # Draw rectangle around the crowd in each frame of the saved clip
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.putText(frame, 'Crowd Detection', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)
                        writer_output.write(frame)
                writer_output.release()
                crowd_detected = False

        # Normalize and apply colormap to heatmap
        heatmap_normalized = heatmap / np.max(heatmap)
        heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Blend heatmap with original image
        img_heatmap = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)

        v = len(groups)  # Number of detected crowds
        cv2.putText(img_heatmap, 'No of Violations : ' + str(v), (15, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 126, 255), 2)
        writer.write(img_heatmap)
        # cv2.imshow("Image", img)
        # cv2.imshow("Images", img_heatmap)
        # cv2.imshow("Imaged", heatmap_colored)
        stack_img = cvzone.stackImages([img, img_heatmap, heatmap_colored], scale=0.8, cols=2)
        cv2.imshow("Images", stack_img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
