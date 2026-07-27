import cv2
import os
import numpy as np
from PIL import Image

# ==============================
# PATHS
# ==============================

path = "dataset"

trainer_folder = "trainer"

trainer_file = os.path.join(
    trainer_folder,
    "trainer.yml"
)

cascade_path = (
    "haarcascade/"
    "haarcascade_frontalface_default.xml"
)

# ==============================
# CHECK DATASET
# ==============================

if not os.path.exists(path):

    print("ERROR: dataset folder not found!")

    input("Press Enter to exit...")

    exit()

# ==============================
# FACE RECOGNIZER
# ==============================

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    cascade_path
)

face_samples = []

ids = []

# ==============================
# READ DATASET
# ==============================

for image_name in os.listdir(path):

    if not image_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        continue

    image_path = os.path.join(
        path,
        image_name
    )

    try:

        image = Image.open(
            image_path
        ).convert("L")

        image_numpy = np.array(
            image,
            "uint8"
        )

        # Example:
        # User.1.1.jpg
        #
        # User = 0
        # 1    = internal face ID
        # 1    = sample number

        filename_parts = image_name.split(".")

        if len(filename_parts) < 4:

            print(
                "Skipping invalid file:",
                image_name
            )

            continue

        face_id = int(
            filename_parts[1]
        )

        faces = detector.detectMultiScale(
            image_numpy
        )

        if len(faces) == 0:

            print(
                "No face detected in:",
                image_name
            )

            continue

        for x, y, w, h in faces:

            face_samples.append(
                image_numpy[
                    y:y + h,
                    x:x + w
                ]
            )

            ids.append(
                face_id
            )

    except Exception as error:

        print(
            "Error reading:",
            image_name
        )

        print(
            error
        )

# ==============================
# CHECK SAMPLES
# ==============================

if len(face_samples) == 0:

    print(
        "ERROR: No face samples found!"
    )

    print(
        "Please register the student again."
    )

    input(
        "Press Enter to exit..."
    )

    exit()

# ==============================
# TRAIN MODEL
# ==============================

print(
    f"Training {len(face_samples)} "
    f"face samples..."
)

recognizer.train(
    face_samples,
    np.array(ids)
)

# ==============================
# SAVE MODEL
# ==============================

os.makedirs(
    trainer_folder,
    exist_ok=True
)

recognizer.write(
    trainer_file
)

print(
    "================================"
)

print(
    "MODEL TRAINED SUCCESSFULLY!"
)

print(
    "================================"
)

print(
    "Student Face IDs trained:"
)

print(
    set(ids)
)

print(
    f"Model saved at: {trainer_file}"
)