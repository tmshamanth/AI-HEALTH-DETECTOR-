import cv2
import csv
import os
import sqlite3
from datetime import datetime


# ==========================
# DATABASE FUNCTIONS
# ==========================

def get_student_name(student_id):
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM students WHERE id=?",
        (student_id,)
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return "Unknown"


def get_all_students():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM students")

    students = cursor.fetchall()

    conn.close()

    return students


# ==========================
# GET REGISTERED STUDENTS
# ==========================

all_students = get_all_students()

total_students = len(all_students)

marked_students = set()


# ==========================
# FACE RECOGNIZER
# ==========================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    "trainer/trainer.yml"
)


faceCascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)


font = cv2.FONT_HERSHEY_SIMPLEX


# ==========================
# CAMERA
# ==========================

cam = cv2.VideoCapture(0)

cam.set(3, 640)
cam.set(4, 480)


minW = int(cam.get(3) * 0.1)
minH = int(cam.get(4) * 0.1)


# ==========================
# ATTENDANCE FILE
# ==========================

attendance_file = "attendance.csv"


if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "ID",
            "Name",
            "Date",
            "Time"
        ])


print("Attendance System Started...")


# ==========================
# CAMERA LOOP
# ==========================

while True:

    ret, img = cam.read()

    if not ret:
        break


    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(minW, minH)
    )


    for (x, y, w, h) in faces:


        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )


        if confidence < 50:


            student_name = get_student_name(
                student_id
            )


            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )


            cv2.putText(
                img,
                student_name,
                (x, y-35),
                font,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                img,
                f"ID: {student_id}",
                (x, y-10),
                font,
                0.7,
                (0, 255, 0),
                2
            )


            today = datetime.now().strftime(
                "%Y-%m-%d"
            )


            # Mark only once
            if student_id not in marked_students:


                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as f:


                    writer = csv.writer(f)


                    writer.writerow([
                        student_id,
                        student_name,
                        today,
                        datetime.now().strftime(
                            "%H:%M:%S"
                        )
                    ])


                marked_students.add(
                    student_id
                )


                print(
                    f"{student_name} - Attendance Marked"
                )


        else:


            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )


            cv2.putText(
                img,
                "Unknown",
                (x, y-10),
                font,
                0.8,
                (0, 0, 255),
                2
            )


    # Show present count
    cv2.putText(
        img,
        f"Present: {len(marked_students)} / {total_students}",
        (20, 40),
        font,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.imshow(
        "Face Attendance System",
        img
    )


    # Automatically close when everyone is present
    if (
        len(marked_students) == total_students
        and total_students > 0
    ):

        print("\n==============================")
        print("ATTENDANCE COMPLETED")
        print("==============================")

        print(
            f"Present: {len(marked_students)}"
        )

        print(
            f"Absent: {total_students - len(marked_students)}"
        )

        print(
            f"Total Students: {total_students}"
        )

        cv2.waitKey(2000)

        break


    # Press ESC to stop manually
    if cv2.waitKey(1) & 0xFF == 27:

        break


# ==========================
# FINAL SUMMARY
# ==========================

cam.release()

cv2.destroyAllWindows()


present = len(marked_students)

absent = total_students - present


print("\n==============================")
print("FINAL ATTENDANCE SUMMARY")
print("==============================")

print(
    f"Present : {present}"
)

print(
    f"Absent  : {absent}"
)

print(
    f"Total   : {total_students}"
)

print("==============================")