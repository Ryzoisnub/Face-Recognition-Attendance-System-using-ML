# 🎓 Face Recognition Attendance System (Web-Based)

A web-based Face Recognition Attendance System built using **Flask** and **OpenCV**.  
The system performs real-time face detection and recognition using the **LBPH algorithm** and automatically stores attendance in **date-wise CSV files**.

---

## 🚀 Features

- 🔍 Real-time face detection using Haar Cascade
- 🧠 Face recognition using LBPH (Local Binary Pattern Histogram)
- 🌐 Flask-based web interface
- 📅 Automatic date-wise attendance logging
- 📁 Structured dataset-based supervised training
- 🎨 Clean UI with external CSS styling
- 🗂 Proper Git management with `.gitignore`

---

## 🛠 Tech Stack

- Python
- OpenCV (opencv-contrib-python)
- Flask
- NumPy
- Pandas
- HTML5 + CSS3

---

## 📂 Project Structure

```

face_reco/
│
├── app.py
├── haarcascade_frontalface_default.xml
├── trainer.yml
├── attendance_YYYY-MM-DD.csv
│
├── dataset/
│   ├── Person_1/
│   ├── Person_2/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── .gitignore

```

---

## ⚙️ How It Works

1. The system loads a labeled dataset of facial images.
2. Faces are detected using Haar Cascade.
3. LBPH algorithm trains a face recognition model.
4. The webcam feed is streamed to the browser using Flask.
5. Recognized faces are marked present.
6. Attendance is stored in a file:

```

attendance_YYYY-MM-DD.csv

````

Each file contains:

| Name | Date | Time |
|------|------|------|

---

## ▶️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd face_reco
````

### 2️⃣ Install dependencies

```bash
pip install flask opencv-contrib-python numpy pandas
```

### 3️⃣ Run the application

```bash
python app.py
```

### 4️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 📸 Dataset Requirement

* Dataset must be structured folder-wise.
* Each folder represents one person.
* Minimum 8–10 images per person recommended.

Example:

```
dataset/
├── Student_1/
├── Student_2/
```

---

## 📌 Output

Attendance files are automatically generated daily:

```
attendance_2026-02-10.csv
attendance_2026-02-11.csv
```

---

## 🎤 Academic Explanation

This system demonstrates supervised machine learning for facial recognition using LBPH.
It integrates computer vision with a web-based backend interface to provide automated, persistent attendance tracking.

---

## ⚠️ Limitations

* Accuracy depends on lighting conditions.
* Requires frontal face images.
* Not optimized for large-scale deployment.

---

## 🔮 Future Improvements

* Database integration (SQLite / MySQL)
* User login system
* Cloud deployment
* Mask detection
* Multi-camera support
* Deep learning upgrade (FaceNet / CNN)

---

## 👨‍💻 Author

Developed as an academic machine learning project integrating Computer Vision and Web Technologies.

```

---

If you want, I can now generate:

- 🧠 A stronger “placement-level” version  
- 📊 Architecture diagram section  
- 🏆 Resume bullet points  
- 🌍 Deployment guide (Render / Railway / Docker)  
- 📈 Add screenshots section template  

What level are we going for — college submission or recruiter-ready? 😄
```
