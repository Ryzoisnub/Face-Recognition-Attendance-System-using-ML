import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

dataset_path = "dataset"
cascade_path = "haarcascade_frontalface_default.xml"
model_path = "trainer.yml"

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for folder_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, folder_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = folder_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save(model_path)

attendance = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            name = label_map[id_]
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")

            if name not in [row[0] for row in attendance]:
                attendance.append([name, date, time])

            text = name
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(attendance, columns=["Name", "Date", "Time"])
df.to_csv("attendance.csv", index=False)
