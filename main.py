import cv2
import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.objects = {}  # Dictionary to store tracked objects
        self.id_count = 0  # Counter to assign IDs to new objects

    def update(self, detections):
        new_objects = {}  # Temporary dictionary to store newly detected objects

        for detection in detections:
            x, y, w, h = detection

            centroid = np.array([x + w / 2, y + h / 2])  # Calculate centroid of the detection

            if len(self.objects) == 0:
                # If there are no tracked objects, assign a new ID
                self.objects[self.id_count] = (centroid, (x, y, w, h))
                self.id_count += 1
            else:
                distances = []
                for obj_id, (obj_centroid, _) in self.objects.items():
                    # Calculate Euclidean distance between centroids
                    distance = np.linalg.norm(centroid - obj_centroid)
                    distances.append(distance)

                min_distance = min(distances)
                min_distance_id = list(self.objects.keys())[distances.index(min_distance)]

                if min_distance < 50:  # Adjust the threshold as per your requirements
                    # Assign the same ID if the distance is below the threshold
                    new_objects[min_distance_id] = (centroid, (x, y, w, h))
                else:
                    # Assign a new ID if the distance is above the threshold
                    self.objects[self.id_count] = (centroid, (x, y, w, h))
                    self.id_count += 1

        # Update the tracked objects with the newly detected objects
        for obj_id, obj_data in new_objects.items():
            self.objects[obj_id] = obj_data

        return self.objects

# Usage:

tracker = EuclideanDistTracker()
detections = []  # Initialize the list before the loop

cap = cv2.VideoCapture(r"D:\Noct\PythonProj\Python\OpenCV Projects\Code\highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Object Detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = object_detector.apply(frame)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id, (centroid, bbox) in boxes_ids.items():
        x, y, w, h = bbox
        cv2.putText(frame, str(box_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(30)
    if key == 27:  # Check for 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
