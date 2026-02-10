from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, "dataset")
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, labels, label_map = [], [], {}
current_id = 0

for folder in os.listdir(dataset_path):
    path = os.path.join(dataset_path, folder)
    if not os.path.isdir(path):
        continue
    label_map[current_id] = folder
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        detected = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in detected:
            faces.append(img[y:y+h, x:x+w])
            labels.append(current_id)
    current_id += 1

recognizer.train(faces, np.array(labels))

attendance = []
today = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today}.csv"

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                name = label_map[id_]
                time = datetime.now().strftime("%H:%M:%S")
                if name not in [row[0] for row in attendance]:
                    attendance.append([name, today, time])
                text, color = name, (0,255,0)
            else:
                text, color = "Unknown", (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,text,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save')
def save():
    df = pd.DataFrame(attendance, columns=["Name", "Date", "Time"])
    if os.path.exists(attendance_file):
        old = pd.read_csv(attendance_file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Name"])
    df.to_csv(attendance_file, index=False)
    return f"Attendance saved as {attendance_file}"

if __name__ == "__main__":
    app.run(debug=True)
