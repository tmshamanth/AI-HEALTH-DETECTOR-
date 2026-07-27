import os
import csv
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")

STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.csv")
ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_DIR,
    "attendance.csv"
)



# =========================================================
# CREATE FOLDERS
# =========================================================

for folder in [
    DATASET_DIR,
    TRAINER_DIR,
    ATTENDANCE_DIR
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


# =========================================================
# CREATE STUDENTS CSV
# =========================================================

if not os.path.exists(STUDENTS_CSV):

    with open(
        STUDENTS_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "reg no",
                "name",
                "folder"
            ]
        )


# =========================================================
# CREATE ATTENDANCE CSV
# =========================================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "roll_no",
                "name",
                "date",
                "time"
            ]
        )


# =========================================================
# FACE DETECTOR
# =========================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================================================
# COLORS
# =========================================================

BG = "#FAF6ED"
GREEN = "#1F4D3A"
GREEN_DEEP = "#163627"
SAGE = "#CFE0D3"
CORAL = "#E85D4C"
INK = "#22291F"
MUTED = "#6E7A6E"
CARD = "#FFFFFF"


# =========================================================
# FONTS
# =========================================================

FONT_TITLE = (
    "Segoe UI",
    22,
    "bold"
)

FONT_SUB = (
    "Segoe UI",
    11
)

FONT_BTN = (
    "Segoe UI",
    12,
    "bold"
)

FONT_LABEL = (
    "Segoe UI",
    11
)


# =========================================================
# CAMERA PREVIEW SIZE
# =========================================================

CAMERA_WIDTH = 700
CAMERA_HEIGHT = 500


# =========================================================
# STYLED BUTTON
# =========================================================

def styled_button(
    parent,
    text,
    command,
    bg=GREEN,
    fg="white",
    width=26
):

    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        font=FONT_BTN,
        relief="flat",
        padx=14,
        pady=10,
        width=width,
        activebackground=GREEN_DEEP,
        activeforeground="white",
        cursor="hand2"
    )


# =========================================================
# BACK BUTTON
# =========================================================

def back_home_button(
    parent,
    controller
):

    return tk.Button(
        parent,
        text="⬅  Back to Home",
        command=lambda:
        controller.show_frame(
            "HomeFrame"
        ),
        bg=CARD,
        fg=GREEN_DEEP,
        font=(
            "Segoe UI",
            10,
            "bold"
        ),
        relief="flat",
        padx=12,
        pady=6,
        cursor="hand2",
        highlightbackground=GREEN,
        highlightthickness=1
    )


# =========================================================
# SCREEN HEADER
# =========================================================

def screen_header(
    parent,
    title,
    controller
):

    bar = tk.Frame(
        parent,
        bg=BG
    )

    bar.pack(
        fill="x",
        padx=24,
        pady=(
            20,
            10
        )
    )

    tk.Label(
        bar,
        text=title,
        font=(
            "Segoe UI",
            18,
            "bold"
        ),
        bg=BG,
        fg=GREEN_DEEP
    ).pack(
        side="left"
    )

    back_home_button(
        bar,
        controller
    ).pack(
        side="right"
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

class AttendanceApp(
    tk.Tk
):

    def __init__(self):

        super().__init__()

        self.title(
            "AI FACE ATTENDANCE"
        )

        self.geometry(
            "1000x750"
        )

        self.configure(
            bg=BG
        )

        self.minsize(
            900,
            700
        )

        container = tk.Frame(
            self,
            bg=BG
        )

        container.pack(
            fill="both",
            expand=True
        )

        self.frames = {}

        for F in (
            HomeFrame,
            RegisterFrame,
            TrainFrame,
            AttendanceFrame,
            ViewAttendanceFrame,
            DeleteFrame
        ):

            frame = F(
                container,
                self
            )

            self.frames[
                F.__name__
            ] = frame

            frame.place(
                relwidth=1,
                relheight=1
            )

        self.show_frame(
            "HomeFrame"
        )

    # -----------------------------------------------------

    def show_frame(
        self,
        name
    ):

        frame = self.frames[name]

        if hasattr(
            frame,
            "on_show"
        ):

            frame.on_show()

        frame.tkraise()


# =========================================================
# HOME PAGE
# =========================================================

class HomeFrame(
    tk.Frame
):

    def __init__(
        self,
        parent,
        controller
    ):

        super().__init__(
            parent,
            bg=BG
        )

        self.controller = controller

        tk.Label(
            self,
            text="AI FACE ATTENDANCE",
            font=FONT_TITLE,
            bg=BG,
            fg=GREEN_DEEP
        ).pack(
            pady=(
                40,
                4
            )
        )

        tk.Label(
            self,
            text="CHOOSE OPTION",
            font=FONT_SUB,
            bg=BG,
            fg=MUTED
        ).pack(
            pady=(
                0,
                30
            )
        )

        menu_card = tk.Frame(
            self,
            bg=CARD,
            padx=30,
            pady=30
        )

        menu_card.pack()

        options = [

            (
                "1.  Add / Register Students",
                lambda:
                controller.show_frame(
                    "RegisterFrame"
                )
            ),

            (
                "2.  Train Faces To AI"
                ""
                "",
                lambda:
                controller.show_frame(
                    "TrainFrame"
                )
            ),

            (
                "3.  Take Attendance",
                lambda:
                controller.show_frame(
                    "AttendanceFrame"
                )
            ),

            (
                "4.  View Attendance",
                lambda:
                controller.show_frame(
                    "ViewAttendanceFrame"
                )
            ),

            (
                "5.  Delete Students from Database",
                lambda:
                controller.show_frame(
                    "DeleteFrame"
                )
            ),

            (
                "6.  Exit",
                self.confirm_exit
            )

        ]

        for text, command in options:

            styled_button(
                menu_card,
                text,
                command
            ).pack(
                pady=6
            )

    def confirm_exit(
        self
    ):

        if messagebox.askyesno(
            "Exit",
            "Close the attendance system?"
        ):

            self.controller.destroy()


# =========================================================
# REGISTER STUDENT
# =========================================================

class RegisterFrame(
    tk.Frame
):

    def __init__(
        self,
        parent,
        controller
    ):

        super().__init__(
            parent,
            bg=BG
        )

        self.controller = controller

        self.cap = None

        self.count = 0

        self.max_images = 40

        self.current_folder = None

        screen_header(
            self,
            "Register Student",
            controller
        )

        form = tk.Frame(
            self,
            bg=BG
        )

        form.pack(
            pady=10
        )

        tk.Label(
            form,
            text="Register Number",
            font=FONT_LABEL,
            bg=BG,
            fg=INK
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=6
        )

        self.roll_entry = tk.Entry(
            form,
            font=FONT_LABEL,
            width=28
        )

        self.roll_entry.grid(
            row=0,
            column=1,
            padx=10
        )

        tk.Label(
            form,
            text="Full Name",
            font=FONT_LABEL,
            bg=BG,
            fg=INK
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=6
        )

        self.name_entry = tk.Entry(
            form,
            font=FONT_LABEL,
            width=28
        )

        self.name_entry.grid(
            row=1,
            column=1,
            padx=10
        )

        # CAMERA FRAME

        camera_frame = tk.Frame(
            self,
            bg="black",
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT
        )

        camera_frame.pack(
            pady=12
        )

        camera_frame.pack_propagate(
            False
        )

        self.video_label = tk.Label(
            camera_frame,
            bg="black"
        )

        self.video_label.pack(
            fill="both",
            expand=True
        )

        self.status = tk.Label(
            self,
            text="Fill details, then start capture.",
            font=FONT_LABEL,
            bg=BG,
            fg=MUTED
        )

        self.status.pack()

        btns = tk.Frame(
            self,
            bg=BG
        )

        btns.pack(
            pady=10
        )

        styled_button(
            btns,
            "Start Face Capture",
            self.start_capture,
            width=20
        ).grid(
            row=0,
            column=0,
            padx=6
        )

        styled_button(
            btns,
            "Stop",
            self.stop_capture,
            bg=CORAL,
            width=12
        ).grid(
            row=0,
            column=1,
            padx=6
        )

    # -----------------------------------------------------

    def start_capture(
        self
    ):

        roll = self.roll_entry.get().strip()

        name = self.name_entry.get().strip()

        if not roll or not name:

            messagebox.showwarning(
                "Missing Information",
                "Enter both register number and name."
            )

            return

        self.current_folder = os.path.join(
            DATASET_DIR,
            f"{roll}_{name}".replace(
                " ",
                "_"
            )
        )

        os.makedirs(
            self.current_folder,
            exist_ok=True
        )

        self.count = 0

        self.cap = cv2.VideoCapture(
            0
        )

        if not self.cap.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Unable to open camera."
            )

            return

        self.status.config(
            text="Capturing faces... Look at the camera.",
            fg=GREEN_DEEP
        )

        self._loop_capture(
            roll,
            name
        )

    # -----------------------------------------------------

    def _loop_capture(
        self,
        roll,
        name
    ):

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):

            return

        ret, frame = self.cap.read()

        if ret:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (
                x,
                y,
                w,
                h
            ) in faces:

                if self.count < self.max_images:

                    self.count += 1

                    face_img = gray[
                        y:y + h,
                        x:x + w
                    ]

                    cv2.imwrite(
                        os.path.join(
                            self.current_folder,
                            f"{self.count}.jpg"
                        ),
                        face_img
                    )

                cv2.rectangle(
                    frame,
                    (
                        x,
                        y
                    ),
                    (
                        x + w,
                        y + h
                    ),
                    (
                        31,
                        77,
                        58
                    ),
                    2
                )

            self._show_frame(
                frame
            )

            self.status.config(
                text=
                f"Capturing face samples: "
                f"{self.count}/{self.max_images}"
            )

            if self.count >= self.max_images:

                self._finish(
                    roll,
                    name
                )

                return

        self.after(
            20,
            lambda:
            self._loop_capture(
                roll,
                name
            )
        )

    # -----------------------------------------------------

    def _show_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        image = image.resize(
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            Image.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.configure(
            image=photo
        )

        self.video_label.image = photo

    # -----------------------------------------------------

    def _finish(
        self,
        roll,
        name
    ):

        self.stop_capture()

        with open(
            STUDENTS_CSV,
            "a",
            newline=""
        ) as file:

            csv.writer(
                file
            ).writerow(
                [
                    roll,
                    name,
                    os.path.basename(
                        self.current_folder
                    )
                ]
            )

        self.status.config(
            text=
            f"Saved {self.count} images for {name}.",
            fg=GREEN_DEEP
        )

        messagebox.showinfo(
            "Registered",
            f"{name} ({roll}) registered successfully."
        )

        self.roll_entry.delete(
            0,
            "end"
        )

        self.name_entry.delete(
            0,
            "end"
        )

    # -----------------------------------------------------

    def stop_capture(
        self
    ):

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        self.video_label.configure(
            image=""
        )

        self.video_label.image = None

        self.status.config(
            text="Camera stopped.",
            fg=MUTED
        )


# =========================================================
# TRAIN MODEL
# =========================================================

class TrainFrame(
    tk.Frame
):

    def __init__(
        self,
        parent,
        controller
    ):

        super().__init__(
            parent,
            bg=BG
        )

        screen_header(
            self,
            "Train Students' Face to AI",
            controller
        )

        tk.Label(
            self,
            text=
            "This reads every image in dataset/ and trains the recognizer.",
            font=FONT_LABEL,
            bg=BG,
            fg=MUTED
        ).pack(
            pady=10
        )

        self.status = tk.Label(
            self,
            text="Ready to train.",
            font=FONT_LABEL,
            bg=BG,
            fg=GREEN_DEEP
        )

        self.status.pack(
            pady=10
        )

        styled_button(
            self,
            "Train Now",
            self.train_model,
            width=20
        ).pack(
            pady=10
        )

    def train_model(
        self
    ):

        self.status.config(
            text="Training model... Please wait.",
            fg=GREEN_DEEP
        )

        self.update_idletasks()

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []

        labels = []

        label_map = {}

        next_label = 0

        for folder in sorted(
            os.listdir(
                DATASET_DIR
            )
        ):

            folder_path = os.path.join(
                DATASET_DIR,
                folder
            )

            if not os.path.isdir(
                folder_path
            ):

                continue

            if "_" not in folder:

                continue

            roll, name = folder.split(
                "_",
                1
            )

            label_map[
                next_label
            ] = (
                roll,
                name
            )

            for img_file in os.listdir(
                folder_path
            ):

                img_path = os.path.join(
                    folder_path,
                    img_file
                )

                img = cv2.imread(
                    img_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if img is None:

                    continue

                faces.append(
                    img
                )

                labels.append(
                    next_label
                )

            next_label += 1

        if not faces:

            self.status.config(
                text="No face data found.",
                fg=CORAL
            )

            messagebox.showwarning(
                "No Data",
                "Register at least one student first."
            )

            return

        recognizer.train(
            faces,
            np.array(
                labels
            )
        )

        recognizer.save(
            TRAINER_FILE
        )

        with open(
            LABELS_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "label",
                    "roll_no",
                    "name"
                ]
            )

            for label, (
                roll,
                name
            ) in label_map.items():

                writer.writerow(
                    [
                        label,
                        roll,
                        name
                    ]
                )

        self.status.config(
            text=
            f"Training completed successfully! "
            f"{len(label_map)} students trained.",
            fg=GREEN_DEEP
        )

        messagebox.showinfo(
            "Training Complete",
            "Face recognition model trained successfully."
        )


# =========================================================
# ATTENDANCE
# =========================================================

class AttendanceFrame(
    tk.Frame
):

    def __init__(
        self,
        parent,
        controller
    ):

        super().__init__(
            parent,
            bg=BG
        )

        self.cap = None

        self.recognizer = None

        self.label_map = {}

        self.marked_today = set()

        screen_header(
            self,
            "Take Attendance",
            controller
        )

        # LARGE CAMERA FRAME

        camera_frame = tk.Frame(
            self,
            bg="black",
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT
        )

        camera_frame.pack(
            pady=12
        )

        camera_frame.pack_propagate(
            False
        )

        self.video_label = tk.Label(
            camera_frame,
            bg="black"
        )

        self.video_label.pack(
            fill="both",
            expand=True
        )

        self.status = tk.Label(
            self,
            text=
            "Press Start Camera to begin recognizing faces.",
            font=FONT_LABEL,
            bg=BG,
            fg=MUTED
        )

        self.status.pack()

        btns = tk.Frame(
            self,
            bg=BG
        )

        btns.pack(
            pady=10
        )

        styled_button(
            btns,
            "Start Camera",
            self.start_camera,
            width=16
        ).grid(
            row=0,
            column=0,
            padx=6
        )

        styled_button(
            btns,
            "Stop",
            self.stop_camera,
            bg=CORAL,
            width=12
        ).grid(
            row=0,
            column=1,
            padx=6
        )

    # -----------------------------------------------------

    def start_camera(
        self
    ):

        if not os.path.exists(
            TRAINER_FILE
        ):

            messagebox.showwarning(
                "Not Trained",
                "Train the model first."
            )

            return

        if not os.path.exists(
            LABELS_FILE
        ):

            messagebox.showwarning(
                "Labels Missing",
                "Train the model again."
            )

            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.recognizer.read(
            TRAINER_FILE
        )

        self.label_map = {}

        with open(
            LABELS_FILE,
            newline=""
        ) as file:

            reader = csv.reader(
                file
            )

            next(
                reader,
                None
            )

            for row in reader:

                if len(row) >= 3:

                    label = int(
                        row[0]
                    )

                    roll = row[1]

                    name = row[2]

                    self.label_map[
                        label
                    ] = (
                        roll,
                        name
                    )

        self.marked_today = set()

        self.cap = cv2.VideoCapture(
            0
        )

        if not self.cap.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Unable to open camera."
            )

            return

        self.status.config(
            text="Camera started. Recognizing faces...",
            fg=GREEN_DEEP
        )

        self._loop_recognize()

    # -----------------------------------------------------

    def _loop_recognize(
        self
    ):

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):

            return

        ret, frame = self.cap.read()

        if ret:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (
                x,
                y,
                w,
                h
            ) in faces:

                label, confidence = self.recognizer.predict(
                    gray[
                        y:y + h,
                        x:x + w
                    ]
                )

                if (
                    confidence < 70
                    and label in self.label_map
                ):

                    roll, name = self.label_map[
                        label
                    ]

                    box_color = (
                        31,
                        77,
                        58
                    )

                    display = (
                        f"{name} | {roll}"
                    )

                    self._mark_attendance(
                        roll,
                        name
                    )

                else:

                    box_color = (
                        232,
                        93,
                        76
                    )

                    display = "Unknown"

                cv2.rectangle(
                    frame,
                    (
                        x,
                        y
                    ),
                    (
                        x + w,
                        y + h
                    ),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    display,
                    (
                        x,
                        y - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

            self._show_frame(
                frame
            )

        self.after(
            20,
            self._loop_recognize
        )

    # -----------------------------------------------------

    def _mark_attendance(
        self,
        roll,
        name
    ):

        today = datetime.date.today().isoformat()

        key = (
            roll,
            today
        )

        if key in self.marked_today:

            return

        self.marked_today.add(
            key
        )

        now = datetime.datetime.now().strftime(
            "%H:%M:%S"
        )

        with open(
            ATTENDANCE_FILE,
            "a",
            newline=""
        ) as file:

            csv.writer(
                file
            ).writerow(
                [
                    roll,
                    name,
                    today,
                    now
                ]
            )

        self.status.config(
            text=
            f"Attendance marked: {name} ({roll})",
            fg=GREEN_DEEP
        )

    # -----------------------------------------------------

    def _show_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        image = image.resize(
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            Image.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.configure(
            image=photo
        )

        self.video_label.image = photo

    # -----------------------------------------------------

    def stop_camera(
        self
    ):

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        self.video_label.configure(
            image=""
        )

        self.video_label.image = None

        self.status.config(
            text="Camera stopped.",
            fg=MUTED
        )

    # -----------------------------------------------------

    def on_show(
        self
    ):

        self.marked_today = set()


# =========================================================
# VIEW ATTENDANCE
# =========================================================

# =========================================================
# VIEW ATTENDANCE
# =========================================================

class ViewAttendanceFrame(tk.Frame):

    def __init__(self, parent, controller):

        super().__init__(
            parent,
            bg=BG
        )

        self.controller = controller

        screen_header(
            self,
            "View Attendance",
            controller
        )

        columns = (
            "register number",
            "name",
            "date",
            "time"
        )

        self.tree = ttk.Treeview(
            self,
            columns=columns,
            show="headings",
            height=16
        )

        for column in columns:

            self.tree.heading(
                column,
                text=column.replace(
                    "_",
                    " "
                ).title()
            )

            self.tree.column(
                column,
                width=180,
                anchor="center"
            )

        self.tree.pack(
            padx=24,
            pady=10,
            fill="both",
            expand=True
        )

        # =========================================
        # CLEAR ATTENDANCE BUTTON
        # =========================================

        styled_button(
            self,
            "🗑️ CLEAR ATTENDANCE LIST",
            self.clear_attendance,
            bg=CORAL,
            width=25
        ).pack(
            pady=12
        )

    # =====================================================
    # LOAD ATTENDANCE
    # =====================================================

    def on_show(self):

        for row in self.tree.get_children():

            self.tree.delete(
                row
            )

        if not os.path.exists(
            ATTENDANCE_FILE
        ):

            return

        with open(
            ATTENDANCE_FILE,
            newline=""
        ) as file:

            reader = csv.reader(
                file
            )

            next(
                reader,
                None
            )

            for row in reader:

                if len(row) == 4:

                    self.tree.insert(
                        "",
                        "end",
                        values=row
                    )

    # =====================================================
    # CLEAR ATTENDANCE LIST
    # =====================================================

    def clear_attendance(self):

        confirm = messagebox.askyesno(
            "Clear Attendance",
            "Are you sure you want to clear the complete attendance list?"
        )

        if not confirm:

            return

        # Recreate attendance CSV with only header

        with open(
            ATTENDANCE_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "roll_no",
                    "name",
                    "date",
                    "time"
                ]
            )

        # Clear the table immediately

        for row in self.tree.get_children():

            self.tree.delete(
                row
            )

        messagebox.showinfo(
            "Attendance Cleared",
            "Attendance list cleared successfully."
        )

# =========================================================
# DELETE STUDENT
# =========================================================

class DeleteFrame(
    tk.Frame
):

    def __init__(
        self,
        parent,
        controller
    ):

        super().__init__(
            parent,
            bg=BG
        )

        screen_header(
            self,
            "Delete Student from Database",
            controller
        )

        self.listbox = tk.Listbox(
            self,
            font=FONT_LABEL,
            width=50,
            height=14
        )

        self.listbox.pack(
            padx=24,
            pady=10
        )

        styled_button(
            self,
            "Delete Selected",
            self.delete_selected,
            bg=CORAL,
            width=20
        ).pack(
            pady=10
        )

        self.students = []

    def on_show(
        self
    ):

        self.listbox.delete(
            0,
            "end"
        )

        self.students = []

        if not os.path.exists(
            STUDENTS_CSV
        ):

            return

        with open(
            STUDENTS_CSV
        ) as file:

            reader = csv.reader(
                file
            )

            next(
                reader,
                None
            )

            for row in reader:

                if not row:

                    continue

                self.students.append(
                    row
                )

                self.listbox.insert(
                    "end",
                    f"{row[0]} - {row[1]}"
                )

    def delete_selected(
        self
    ):

        selected = self.listbox.curselection()

        if not selected:

            messagebox.showwarning(
                "No Selection",
                "Select a student to delete."
            )

            return

        roll, name, folder = self.students[
            selected[0]
        ]

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete {name} ({roll})?"
        ):

            return

        folder_path = os.path.join(
            DATASET_DIR,
            folder
        )

        if os.path.isdir(
            folder_path
        ):

            for file_name in os.listdir(
                folder_path
            ):

                os.remove(
                    os.path.join(
                        folder_path,
                        file_name
                    )
                )

            os.rmdir(
                folder_path
            )

        remaining = [
            row
            for row in self.students
            if row[0] != roll
        ]

        with open(
            STUDENTS_CSV,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "roll_no",
                    "name",
                    "folder"
                ]
            )

            writer.writerows(
                remaining
            )

        messagebox.showinfo(
            "Deleted",
            f"{name} removed successfully.\n\n"
            "Please train the model again."
        )

        self.on_show()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app = AttendanceApp()

    app.mainloop()