import cv2
import sqlite3
import os
import sys

# ==========================
# GET DETAILS FROM GUI
# ==========================

if len(sys.argv) < 3:
    print("Student details not received.")
    exit()

student_id = sys.argv[1]
student_name = sys.argv[2]


# ==========================
# DATABASE
# ==========================

conn = sqlite3.connect("students.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
)
""")

cursor.execute(
    "INSERT OR REPLACE INTO students(id, name) VALUES (?, ?)",
    (student_id, student_name)
)

conn.commit()
conn.close()


# ==========================
# DATASET FOLDER
# ==========================

if not os.path.exists("dataset"):
    os.makedirs("dataset")


# ==========================
# FACE DETECTOR
# ==========================

face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print("Look at the camera...")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        filename = f"dataset/User.{student_id}.{count}.jpg"

        cv2.imwrite(
            filename,
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/100",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Register Student - Look at Camera", frame)

    key = cv2.waitKey(100) & 0xff

    if key == 27:
        break

    if count >= 100:
        break


camera.release()
cv2.destroyAllWindows()

print("Student registered successfully!")