from __future__ import annotations

import contextlib
import csv
import datetime
import hashlib
import json
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Iterable

import customtkinter as ctk
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




# ==============================================================================
# SECTION 1 — UI THEME: color palette + animation helpers
# ==============================================================================

PALETTE = {
    "dark": {
        "bg": "#0f1115",
        "sidebar": "#15181f",
        "card": "#1c2029",
        "card_alt": "#20242e",
        "accent": "#4f8cff",
        "accent_hover": "#3d70dd",
        "accent_soft": "#26314a",
        "text": "#eef1f7",
        "text_muted": "#93a0b4",
        "success": "#37d67a",
        "warning": "#ffb443",
        "danger": "#ef5461",
        "danger_hover": "#c9414c",
    },
    "light": {
        "bg": "#f3f5f9",
        "sidebar": "#ffffff",
        "card": "#ffffff",
        "card_alt": "#eef1f7",
        "accent": "#3f6df0",
        "accent_hover": "#2f57cf",
        "accent_soft": "#e2e9fd",
        "text": "#181b22",
        "text_muted": "#5b6472",
        "success": "#1f9d5c",
        "warning": "#c97b12",
        "danger": "#d0384a",
        "danger_hover": "#ad2c3b",
    },
}

FONT_FAMILY = "Segoe UI"


def ease_out_cubic(t: float) -> float:
    """Standard ease-out cubic, t in [0, 1]."""
    return 1 - pow(1 - t, 3)


def animate(widget, duration_ms: int, on_step, on_done=None, fps: int = 60):
    """Drive a generic animation with easing over `duration_ms` milliseconds.

    `on_step(progress)` is called every frame with an eased progress value
    in [0, 1]. `on_done` is called once at the end. The animation is driven
    entirely through `widget.after(...)`, so it never blocks the mainloop.
    """
    steps = max(1, int(duration_ms / (1000 / fps)))
    interval = int(1000 / fps)

    state = {"i": 0}

    def _tick():
        try:
            if not widget.winfo_exists():
                return
            state["i"] += 1
            t = min(1.0, state["i"] / steps)
            on_step(ease_out_cubic(t))
            if t < 1.0:
                widget.after(interval, _tick)
            elif on_done:
                on_done()
        except tk.TclError:
            # Widget was destroyed mid-animation (e.g. rapid page switch or
            # a new toast replacing this one) -- stop quietly.
            pass

    widget.after(interval, _tick)


def animate_progressbar(bar, target_value: float, duration_ms: int = 500):
    """Smoothly animate a CTkProgressBar from its current value to target."""
    try:
        start_value = bar.get()
    except Exception:
        start_value = 0.0

    def _step(progress):
        bar.set(start_value + (target_value - start_value) * progress)

    animate(bar, duration_ms, _step)


def animate_counter(label, target_value: int, duration_ms: int = 600, suffix: str = ""):
    """Smoothly count a CTkLabel's displayed number up to target_value."""

    def _step(progress):
        current = int(round(target_value * progress))
        label.configure(text=f"{current}{suffix}")

    animate(label, duration_ms, _step)


def slide_in(frame, direction: str = "right", duration_ms: int = 260):
    """Slide a `.place()`-managed frame into view from off-screen.

    `direction` is where the frame slides in *from* ("right" or "left").
    The frame must already be placed with relwidth=1, relheight=1.
    """
    start_x = 1.0 if direction == "right" else -1.0

    def _step(progress):
        current_x = start_x * (1 - progress)
        frame.place(relx=current_x, rely=0, relwidth=1, relheight=1)

    animate(frame, duration_ms, _step)

# ==============================================================================
# SECTION 2 — TOOLTIP: lightweight hover tooltip widget
# ==============================================================================

class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#22252c",
            foreground="#eef1f7",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            padx=8,
            pady=4,
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None

# ==============================================================================
# SECTION 3 — DATABASE: SQLite prediction history storage
# ==============================================================================

DB_PATH = Path(__file__).with_name("prediction_history.db")


class HistoryDatabase:
    """Manages all reads/writes to the prediction history table."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symptoms TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    date_time TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add_record(self, symptoms: Iterable[str], disease: str, confidence: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO history (symptoms, disease, confidence, date_time)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ", ".join(symptoms),
                    disease,
                    confidence,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def fetch_all(self, search: str | None = None) -> list[tuple]:
        """Return records newest-first, optionally filtered by a search term
        matched against the disease name or symptom list (case-insensitive)."""
        with self._connect() as conn:
            if search:
                like = f"%{search.lower()}%"
                cursor = conn.execute(
                    """
                    SELECT id, symptoms, disease, confidence, date_time
                    FROM history
                    WHERE LOWER(disease) LIKE ? OR LOWER(symptoms) LIKE ?
                    ORDER BY id DESC
                    """,
                    (like, like),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, symptoms, disease, confidence, date_time
                    FROM history
                    ORDER BY id DESC
                    """
                )
            return cursor.fetchall()

    def delete_record(self, record_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (record_id,))

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")

    def export_csv(self, dest_path: Path) -> Path:
        rows = self.fetch_all()
        with open(dest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ID", "Symptoms", "Disease", "Confidence (%)", "Date/Time"])
            writer.writerows(rows)
        return dest_path

# ==============================================================================
# SECTION 4 — ML MODEL: dataset, training, evaluation, persistence
# ==============================================================================

MODEL_DIR = Path(__file__).with_name("model_artifacts")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
META_PATH = MODEL_DIR / "meta.json"

SYMPTOMS = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Vomiting",
    "Abdominal Pain",
    "Skin Rash",
    "Chest Pain",
    "Sore Throat",
    "Shortness of Breath",
    "Runny Nose",
    "Sneezing",
    "Body Aches",
    "Chills",
    "Diarrhea",
    "Nausea",
    "Loss of Appetite",
    "Dizziness",
    "Joint Pain",
    "Muscle Pain",
    "Swelling",
    "Itching",
    "Ear Pain",
    "Loss of Smell or Taste",
    "Difficulty Swallowing",
    "Frequent Urination",
    "Burning Urination",
    "Back Pain",
    "Unexplained Weight Loss",
    "Excessive Thirst",
    "Palpitations",
    "Night Sweats",
    "Jaundice",
    "Wheezing",
    "Red or Watery Eyes",
    "Swollen Lymph Nodes",
]

# Baseline "background noise" probability applied to every symptom a disease
# profile doesn't explicitly call out below, so generated rows aren't
# perfectly clean 0/1 patterns.
_BASELINE_PROB = 0.04


def _full_profile(overrides: dict[str, float]) -> dict[str, float]:
    """Fill in the baseline probability for every symptom not explicitly
    listed in `overrides`, so each disease profile below only needs to name
    the symptoms that are actually notable for it."""
    return {symptom: overrides.get(symptom, _BASELINE_PROB) for symptom in SYMPTOMS}


# ---------------------------------------------------------------------------
# Symptom -> disease profiles used to *generate* a larger synthetic dataset.
# Each profile is the "typical" symptom pattern for that disease; individual
# rows are produced by flipping bits with a small amount of noise so the
# dataset has enough size/variety for a meaningful train/test split.
#
# IMPORTANT: this is a broad, educational set covering common conditions
# across major body systems (respiratory, GI, cardiovascular, dermatological,
# infectious, endocrine, musculoskeletal, ENT, urinary) -- it is NOT an
# exhaustive worldwide medical database. Real clinical diagnosis involves
# thousands of conditions, lab tests, imaging, and history that a symptom
# checklist alone can never capture; treat every prediction here as a
# rough educational guess, never a diagnosis.
# ---------------------------------------------------------------------------
_DISEASE_PROFILES: dict[str, dict[str, float]] = {
    "Flu (Influenza)": _full_profile({
        "Fever": 0.9, "Cough": 0.8, "Headache": 0.6, "Fatigue": 0.9,
        "Body Aches": 0.8, "Chills": 0.7, "Sore Throat": 0.5,
        "Runny Nose": 0.3, "Loss of Appetite": 0.4,
    }),
    "Common Cold": _full_profile({
        "Cough": 0.7, "Runny Nose": 0.8, "Sneezing": 0.8, "Sore Throat": 0.75,
        "Headache": 0.4, "Fatigue": 0.35, "Fever": 0.2,
    }),
    "COVID-19": _full_profile({
        "Fever": 0.7, "Cough": 0.7, "Fatigue": 0.7, "Loss of Smell or Taste": 0.6,
        "Sore Throat": 0.4, "Shortness of Breath": 0.4, "Body Aches": 0.5,
        "Headache": 0.5, "Chills": 0.3,
    }),
    "Food Poisoning": _full_profile({
        "Vomiting": 0.85, "Abdominal Pain": 0.9, "Diarrhea": 0.8,
        "Nausea": 0.8, "Fever": 0.4, "Fatigue": 0.5, "Chills": 0.2,
    }),
    "Skin Allergy": _full_profile({
        "Skin Rash": 0.9, "Itching": 0.85, "Swelling": 0.35,
        "Red or Watery Eyes": 0.2,
    }),
    "Heart Problem": _full_profile({
        "Chest Pain": 0.9, "Shortness of Breath": 0.75, "Fatigue": 0.6,
        "Palpitations": 0.7, "Dizziness": 0.4, "Swelling": 0.35,
    }),
    "Migraine": _full_profile({
        "Headache": 0.95, "Dizziness": 0.4, "Nausea": 0.5,
        "Loss of Appetite": 0.3, "Red or Watery Eyes": 0.25,
    }),
    "Asthma": _full_profile({
        "Shortness of Breath": 0.85, "Wheezing": 0.85, "Cough": 0.6,
        "Chest Pain": 0.3,
    }),
    "Bronchitis": _full_profile({
        "Cough": 0.9, "Chest Pain": 0.3, "Fatigue": 0.5, "Wheezing": 0.4,
        "Shortness of Breath": 0.3, "Fever": 0.3,
    }),
    "Pneumonia": _full_profile({
        "Fever": 0.8, "Cough": 0.85, "Shortness of Breath": 0.7,
        "Chest Pain": 0.6, "Fatigue": 0.7, "Chills": 0.5,
    }),
    "Gastroenteritis": _full_profile({
        "Diarrhea": 0.9, "Vomiting": 0.7, "Abdominal Pain": 0.7,
        "Nausea": 0.7, "Fever": 0.3, "Fatigue": 0.4,
    }),
    "Urinary Tract Infection": _full_profile({
        "Frequent Urination": 0.85, "Burning Urination": 0.85,
        "Abdominal Pain": 0.4, "Fever": 0.25, "Back Pain": 0.3,
    }),
    "Sinusitis": _full_profile({
        "Headache": 0.6, "Runny Nose": 0.6, "Sneezing": 0.3, "Sore Throat": 0.3,
        "Fever": 0.2, "Loss of Smell or Taste": 0.3,
    }),
    "Strep Throat": _full_profile({
        "Sore Throat": 0.9, "Fever": 0.6, "Difficulty Swallowing": 0.6,
        "Swollen Lymph Nodes": 0.5, "Headache": 0.3,
    }),
    "Conjunctivitis (Pink Eye)": _full_profile({
        "Red or Watery Eyes": 0.9, "Itching": 0.5, "Swelling": 0.3,
    }),
    "Chickenpox": _full_profile({
        "Skin Rash": 0.9, "Fever": 0.6, "Itching": 0.7, "Fatigue": 0.4,
        "Headache": 0.3,
    }),
    "Measles": _full_profile({
        "Skin Rash": 0.85, "Fever": 0.8, "Cough": 0.5, "Runny Nose": 0.5,
        "Red or Watery Eyes": 0.5, "Fatigue": 0.5,
    }),
    "Dengue Fever": _full_profile({
        "Fever": 0.9, "Body Aches": 0.8, "Headache": 0.7, "Joint Pain": 0.7,
        "Skin Rash": 0.4, "Fatigue": 0.6, "Nausea": 0.4,
    }),
    "Malaria": _full_profile({
        "Fever": 0.9, "Chills": 0.85, "Body Aches": 0.6, "Fatigue": 0.7,
        "Headache": 0.5, "Nausea": 0.4, "Vomiting": 0.3,
    }),
    "Typhoid Fever": _full_profile({
        "Fever": 0.9, "Fatigue": 0.6, "Abdominal Pain": 0.5, "Headache": 0.5,
        "Loss of Appetite": 0.5, "Diarrhea": 0.3, "Body Aches": 0.3,
    }),
    "Hypertension": _full_profile({
        "Headache": 0.4, "Dizziness": 0.4, "Chest Pain": 0.2,
        "Palpitations": 0.3, "Fatigue": 0.3,
    }),
    "Type 2 Diabetes": _full_profile({
        "Excessive Thirst": 0.85, "Frequent Urination": 0.8, "Fatigue": 0.6,
        "Unexplained Weight Loss": 0.5, "Dizziness": 0.2, "Loss of Appetite": 0.2,
    }),
    "Hypothyroidism": _full_profile({
        "Fatigue": 0.8, "Muscle Pain": 0.4, "Joint Pain": 0.3,
        "Dizziness": 0.2, "Chills": 0.3,
    }),
    "Gastric Ulcer / GERD": _full_profile({
        "Abdominal Pain": 0.8, "Nausea": 0.5, "Loss of Appetite": 0.4,
        "Chest Pain": 0.4, "Vomiting": 0.3,
    }),
    "Kidney Stones": _full_profile({
        "Back Pain": 0.85, "Abdominal Pain": 0.6, "Burning Urination": 0.5,
        "Nausea": 0.4, "Vomiting": 0.3, "Frequent Urination": 0.3,
    }),
    "Arthritis": _full_profile({
        "Joint Pain": 0.9, "Swelling": 0.5, "Muscle Pain": 0.3,
        "Fatigue": 0.3, "Back Pain": 0.3,
    }),
    "Eczema (Dermatitis)": _full_profile({
        "Skin Rash": 0.85, "Itching": 0.85, "Swelling": 0.2,
    }),
    "Ear Infection (Otitis Media)": _full_profile({
        "Ear Pain": 0.9, "Fever": 0.4, "Fatigue": 0.3, "Dizziness": 0.2,
    }),
}

DISEASES = list(_DISEASE_PROFILES.keys())


def _generate_dataset(samples_per_disease: int = 35, seed: int = 42) -> pd.DataFrame:
    """Generate a larger synthetic dataset from the disease profiles above."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for disease, profile in _DISEASE_PROFILES.items():
        for _ in range(samples_per_disease):
            row = {symptom: int(rng.random() < prob) for symptom, prob in profile.items()}
            row["Disease"] = disease
            rows.append(row)
    df = pd.DataFrame(rows)[SYMPTOMS + ["Disease"]]
    # Shuffle so train_test_split's stratification sees a mixed order.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    label_encoder: LabelEncoder
    accuracy: float
    f1: float
    n_train: int
    n_test: int
    n_features: int
    n_classes: int


def load_or_train_model(force_retrain: bool = False) -> ModelBundle:
    """Load a cached model if the dataset hasn't changed, otherwise train
    (and evaluate) a fresh one and cache it to disk."""

    df = _generate_dataset()
    X = df[SYMPTOMS]
    y = df["Disease"]
    data_hash = _dataset_hash(df)

    if not force_retrain and MODEL_PATH.exists() and ENCODER_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        if meta.get("data_hash") == data_hash:
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            return ModelBundle(
                model=model,
                label_encoder=label_encoder,
                accuracy=meta["accuracy"],
                f1=meta["f1"],
                n_train=meta["n_train"],
                n_test=meta["n_test"],
                n_features=len(SYMPTOMS),
                n_classes=len(label_encoder.classes_),
            )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Refit on the full dataset for the model that actually ships, now that
    # we have an honest held-out score from the split above.
    model.fit(X, y_encoded)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "data_hash": data_hash,
                "accuracy": accuracy,
                "f1": f1,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
    )

    return ModelBundle(
        model=model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        f1=f1,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(SYMPTOMS),
        n_classes=len(label_encoder.classes_),
    )


def predict_top_n(bundle: ModelBundle, selected_symptoms: set[str], top_n: int = 3):
    """Return a list of (disease_name, probability_percent) sorted descending,
    for the given set of selected symptom names."""
    input_row = {symptom: int(symptom in selected_symptoms) for symptom in SYMPTOMS}
    input_df = pd.DataFrame([input_row], columns=SYMPTOMS)

    probabilities = bundle.model.predict_proba(input_df)[0]
    class_names = bundle.label_encoder.inverse_transform(range(len(probabilities)))

    ranked = sorted(zip(class_names, probabilities), key=lambda pair: pair[1], reverse=True)
    return [(name, prob * 100) for name, prob in ranked[:top_n]]

# ==============================================================================
# SECTION 5 — GUI: CustomTkinter application
# ==============================================================================

SYMPTOM_INFO = {
    "Fever": "Elevated body temperature, often a sign of infection.",
    "Cough": "Reflex action to clear the airway; dry or productive.",
    "Headache": "Pain or pressure in the head or upper neck.",
    "Fatigue": "Persistent tiredness not relieved by rest.",
    "Vomiting": "Forceful expulsion of stomach contents.",
    "Abdominal Pain": "Discomfort or ache anywhere in the belly area.",
    "Skin Rash": "Change in skin color, texture, or appearance.",
    "Chest Pain": "Discomfort or pressure in the chest area.",
    "Sore Throat": "Pain, scratchiness, or irritation in the throat.",
    "Shortness of Breath": "Difficulty breathing or feeling breathless.",
    "Runny Nose": "Excess nasal discharge, often clear or watery.",
    "Sneezing": "Repeated, involuntary expulsion of air through the nose.",
    "Body Aches": "General, widespread muscular soreness or discomfort.",
    "Chills": "Feeling cold and shivering, often alongside a fever.",
    "Diarrhea": "Frequent, loose, or watery bowel movements.",
    "Nausea": "An uneasy stomach sensation, often preceding vomiting.",
    "Loss of Appetite": "Reduced desire to eat.",
    "Dizziness": "A spinning, lightheaded, or unsteady sensation.",
    "Joint Pain": "Pain, stiffness, or aching in one or more joints.",
    "Muscle Pain": "Soreness or aching localized to muscle tissue.",
    "Swelling": "Visible puffiness or enlargement of a body area.",
    "Itching": "An irritating sensation prompting the urge to scratch.",
    "Ear Pain": "Discomfort or ache inside or around the ear.",
    "Loss of Smell or Taste": "Reduced or absent sense of smell or taste.",
    "Difficulty Swallowing": "Pain or trouble moving food/liquid down the throat.",
    "Frequent Urination": "Needing to urinate more often than usual.",
    "Burning Urination": "A stinging or burning sensation while urinating.",
    "Back Pain": "Pain anywhere along the upper, middle, or lower back.",
    "Unexplained Weight Loss": "Losing weight without dieting or exercising more.",
    "Excessive Thirst": "Feeling unusually or persistently thirsty.",
    "Palpitations": "A sensation of a racing, pounding, or skipped heartbeat.",
    "Night Sweats": "Excessive sweating during sleep, unrelated to room temperature.",
    "Jaundice": "Yellowing of the skin or the whites of the eyes.",
    "Wheezing": "A high-pitched whistling sound while breathing.",
    "Red or Watery Eyes": "Eyes that appear red, irritated, or teary.",
    "Swollen Lymph Nodes": "Tender or enlarged glands, often in the neck.",
}

NAV_ITEMS = [
    ("dashboard", "🏠", "Dashboard"),
    ("predict", "🔍", "Predict Disease"),
    ("history", "📋", "History"),
]


class DiseasePredictionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AI Disease Prediction System")
        self.geometry("1200x760")
        self.minsize(1000, 640)
        self.resizable(True, True)

        self.mode = "dark"
        self.colors = PALETTE[self.mode]
        ctk.set_appearance_mode(self.mode)
        ctk.set_default_color_theme("blue")

        self.db = HistoryDatabase()

        # Selected symptoms: name -> BooleanVar-like bool flag (plain dict,
        # chips manage their own toggled visual state).
        self.selected_symptoms: dict[str, bool] = {}
        self.chip_buttons: dict[str, ctk.CTkButton] = {}
        self.nav_buttons: dict[str, ctk.CTkButton] = {}

        self.current_page = None
        self._current_frame = None
        self._toast_frame = None

        self._build_shell()

        # Train/load the model once at startup.
        self.bundle = load_or_train_model()

        self.navigate("dashboard")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def report_callback_exception(self, exc, val, tb):
        # CustomTkinter has a known, harmless quirk: if a widget (e.g. a
        # button or chip) is destroyed -- typically while the mouse is
        # still hovering over it, such as during a page-switch animation
        # -- Tkinter's internal "focus follows mouse" binding can still
        # fire and try to call .focus() on that widget's now-gone internal
        # canvas. That raises a TclError("invalid command name ...") from
        # deep inside Tkinter's own callback plumbing, not from our code,
        # and it doesn't affect app state or functionality. We swallow
        # only that specific message here and still print anything else.
        if issubclass(exc, tk.TclError) and "invalid command name" in str(val):
            return
        import traceback

        traceback.print_exception(exc, val, tb)

    # ==================================================================
    # SHELL: sidebar + page container (built once; pages rebuild inside it)
    # ==================================================================

    def _build_shell(self):
        self.configure(fg_color=self.colors["bg"])

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=self.colors["sidebar"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="AI HEALTH", font=(FONT_FAMILY, 25, "bold"), text_color=self.colors["text"]
        ).pack(pady=(40, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="DISEASE PREDICTION",
            font=(FONT_FAMILY, 12),
            text_color=self.colors["text_muted"],
        ).pack(pady=(0, 35))

        for name, icon, label in NAV_ITEMS:
            btn = ctk.CTkButton(
                self.sidebar,
                text=f"{icon}  {label}",
                height=46,
                corner_radius=10,
                anchor="w",
                font=(FONT_FAMILY, 14),
                fg_color="transparent",
                text_color=self.colors["text"],
                hover_color=self.colors["card_alt"],
                command=lambda n=name: self.navigate(n),
            )
            btn.pack(padx=22, pady=6, fill="x")
            self.nav_buttons[name] = btn

        ctk.CTkFrame(self.sidebar, fg_color="transparent", height=1).pack(expand=True, fill="both")

        self.theme_button = ctk.CTkButton(
            self.sidebar,
            text="🌙  Dark Mode",
            height=42,
            corner_radius=10,
            fg_color="transparent",
            border_width=1,
            border_color=self.colors["text_muted"],
            text_color=self.colors["text"],
            hover_color=self.colors["card_alt"],
            command=self.toggle_theme,
        )
        self.theme_button.pack(padx=22, pady=(6, 6), fill="x")

        ctk.CTkButton(
            self.sidebar,
            text="❌  Exit",
            height=42,
            corner_radius=10,
            fg_color=self.colors["danger"],
            hover_color=self.colors["danger_hover"],
            command=self.on_closing,
        ).pack(padx=22, pady=(0, 30), fill="x")

        # Right side: static disclaimer banner + animated page container.
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=self.colors["bg"])
        self.content.pack(side="right", fill="both", expand=True)

        self.banner = ctk.CTkLabel(
            self.content,
            text="⚠  Educational demo only — not a substitute for professional medical advice.",
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=self.colors["accent_soft"],
            text_color=self.colors["warning"],
            corner_radius=8,
            height=34,
        )
        self.banner.pack(fill="x", padx=20, pady=(16, 0))

        self.page_container = ctk.CTkFrame(self.content, corner_radius=0, fg_color=self.colors["bg"])
        self.page_container.pack(fill="both", expand=True)

    def _rebuild_shell(self):
        """Full rebuild used after a theme change so every color updates."""
        for widget in self.winfo_children():
            widget.destroy()
        self.nav_buttons.clear()
        self._current_frame = None
        self._build_shell()
        self.navigate(self.current_page or "dashboard", animate=False)

    def toggle_theme(self):
        self.mode = "light" if self.mode == "dark" else "dark"
        self.colors = PALETTE[self.mode]
        ctk.set_appearance_mode(self.mode)
        self._rebuild_shell()

    # ==================================================================
    # NAVIGATION
    # ==================================================================

    def navigate(self, page_name: str, animate: bool = True):
        order = [n for n, _, _ in NAV_ITEMS]
        direction = "right"
        if self.current_page in order and page_name in order:
            direction = "right" if order.index(page_name) >= order.index(self.current_page) else "left"

        for name, btn in self.nav_buttons.items():
            active = name == page_name
            btn.configure(
                fg_color=self.colors["accent_soft"] if active else "transparent",
                text_color=self.colors["accent"] if active else self.colors["text"],
            )

        new_frame = ctk.CTkFrame(self.page_container, fg_color=self.colors["bg"])
        builder = {
            "dashboard": self._build_dashboard,
            "predict": self._build_prediction_page,
            "history": self._build_history_page,
        }[page_name]
        builder(new_frame)

        old_frame = self._current_frame
        self.current_page = page_name

        if animate and old_frame is not None:
            new_frame.place(relx=1.0 if direction == "right" else -1.0, rely=0, relwidth=1, relheight=1)
            slide_in(new_frame, direction=direction, duration_ms=240)
            self.after(260, lambda: old_frame.destroy() if old_frame.winfo_exists() else None)
        else:
            if old_frame is not None:
                old_frame.destroy()
            new_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._current_frame = new_frame

    # ==================================================================
    # DASHBOARD PAGE
    # ==================================================================

    def _build_dashboard(self, parent):
        ctk.CTkLabel(
            parent, text="AI HEALTH DASHBOARD", font=(FONT_FAMILY, 30, "bold"), text_color=self.colors["text"]
        ).pack(pady=(35, 6))

        ctk.CTkLabel(
            parent,
            text="Machine-learning based disease prediction, with an honest look at model quality.",
            font=(FONT_FAMILY, 15),
            text_color=self.colors["text_muted"],
        ).pack(pady=(0, 30))

        cards_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cards_frame.pack(pady=10)

        cards = [
            ("🤖", "AI MODEL", None, "Random Forest"),
            ("🩺", "SYMPTOMS", len(SYMPTOMS), None),
            ("🧠", "DISEASES", self.bundle.n_classes, None),
            ("🎯", "TEST ACCURACY", round(self.bundle.accuracy * 100), "%"),
            ("📊", "TEST F1 SCORE", round(self.bundle.f1 * 100), "%"),
        ]

        for i, (icon, title, number_value, static_or_suffix) in enumerate(cards):
            card = ctk.CTkFrame(cards_frame, width=195, height=140, corner_radius=14, fg_color=self.colors["card"])
            card.grid(row=0, column=i, padx=10, pady=5)
            card.grid_propagate(False)

            ctk.CTkLabel(card, text=icon, font=(FONT_FAMILY, 28)).pack(pady=(16, 0))
            ctk.CTkLabel(card, text=title, font=(FONT_FAMILY, 12), text_color=self.colors["text_muted"]).pack()

            value_label = ctk.CTkLabel(card, text="0", font=(FONT_FAMILY, 20, "bold"), text_color=self.colors["accent"])
            value_label.pack(pady=6)

            if number_value is None:
                value_label.configure(text=static_or_suffix)
            else:
                suffix = static_or_suffix or ""
                animate_counter(value_label, number_value, duration_ms=700, suffix=suffix)

        info_frame = ctk.CTkFrame(parent, fg_color=self.colors["card"], corner_radius=14)
        info_frame.pack(padx=60, pady=30, fill="x")

        ctk.CTkLabel(
            info_frame,
            text=(
                f"Trained on {self.bundle.n_train} samples, evaluated on {self.bundle.n_test} held-out samples.\n"
                "Numbers above reflect performance on data the model did not train on."
            ),
            font=(FONT_FAMILY, 13),
            text_color=self.colors["text_muted"],
            justify="left",
        ).pack(padx=20, pady=16, anchor="w")

        retrain_btn = ctk.CTkButton(
            parent,
            text="🔁  Retrain Model",
            width=200,
            height=42,
            corner_radius=10,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            command=self._retrain_model,
        )
        retrain_btn.pack(pady=(0, 10))
        ToolTip(retrain_btn, "Regenerates the training data and retrains the\nRandomForest from scratch, then refreshes these stats.")

    def _retrain_model(self):
        self.bundle = load_or_train_model(force_retrain=True)
        self.navigate("dashboard", animate=False)
        self.show_toast("Model retrained successfully", kind="success")

    # ==================================================================
    # PREDICTION PAGE
    # ==================================================================

    def _build_prediction_page(self, parent):
        ctk.CTkLabel(
            parent, text="SELECT YOUR SYMPTOMS", font=(FONT_FAMILY, 26, "bold"), text_color=self.colors["text"]
        ).pack(pady=(25, 4))

        ctk.CTkLabel(
            parent,
            text="Tap the symptoms you're experiencing. Hover a chip for a quick explanation.",
            font=(FONT_FAMILY, 13),
            text_color=self.colors["text_muted"],
        ).pack(pady=(0, 15))

        self.selected_symptoms = {s: False for s in SYMPTOMS}
        self.chip_buttons = {}

        chip_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent", height=320)
        chip_scroll.pack(padx=30, pady=5, fill="both", expand=True)

        columns = 4
        for i, symptom in enumerate(SYMPTOMS):
            chip_scroll.grid_columnconfigure(i % columns, weight=1)
            chip = ctk.CTkButton(
                chip_scroll,
                text=f"☐  {symptom}",
                height=42,
                corner_radius=10,
                fg_color=self.colors["card"],
                text_color=self.colors["text"],
                hover_color=self.colors["card_alt"],
                border_width=1,
                border_color=self.colors["card_alt"],
                anchor="w",
                command=lambda s=symptom: self._toggle_symptom(s),
            )
            chip.grid(row=i // columns, column=i % columns, padx=10, pady=8, sticky="ew")
            ToolTip(chip, SYMPTOM_INFO.get(symptom, symptom))
            self.chip_buttons[symptom] = chip

        self.warning_label = ctk.CTkLabel(parent, text="", font=(FONT_FAMILY, 13, "bold"), text_color=self.colors["danger"])
        self.warning_label.pack(pady=(6, 0))

        button_row = ctk.CTkFrame(parent, fg_color="transparent")
        button_row.pack(pady=18)

        self.predict_button = ctk.CTkButton(
            button_row,
            text="🔍  PREDICT DISEASE",
            width=260,
            height=48,
            corner_radius=12,
            font=(FONT_FAMILY, 15, "bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            command=self._on_predict_clicked,
        )
        self.predict_button.grid(row=0, column=0, padx=8)

        ctk.CTkButton(
            button_row,
            text="🔄  Clear",
            width=140,
            height=48,
            corner_radius=12,
            fg_color="transparent",
            border_width=1,
            border_color=self.colors["text_muted"],
            text_color=self.colors["text"],
            hover_color=self.colors["card_alt"],
            command=lambda: self.navigate("predict", animate=False),
        ).grid(row=0, column=1, padx=8)

        self.results_container = ctk.CTkFrame(parent, fg_color="transparent")
        self.results_container.pack(fill="x", padx=40, pady=(10, 20))

    def _toggle_symptom(self, symptom: str):
        self.selected_symptoms[symptom] = not self.selected_symptoms[symptom]
        chip = self.chip_buttons[symptom]
        if self.selected_symptoms[symptom]:
            chip.configure(
                text=f"☑  {symptom}",
                fg_color=self.colors["accent_soft"],
                text_color=self.colors["accent"],
                border_color=self.colors["accent"],
            )
        else:
            chip.configure(
                text=f"☐  {symptom}",
                fg_color=self.colors["card"],
                text_color=self.colors["text"],
                border_color=self.colors["card_alt"],
            )

    def _on_predict_clicked(self):
        selected = [s for s, is_on in self.selected_symptoms.items() if is_on]

        if not selected:
            self.warning_label.configure(text="⚠  Please select at least one symptom.")
            self.after(1800, lambda: self.warning_label.configure(text=""))
            return

        self.warning_label.configure(text="")
        self.predict_button.configure(state="disabled", text="⏳  Analyzing...")
        # Small artificial delay so the "Analyzing..." state is visibly felt,
        # and the button re-enables itself once results are drawn in.
        self.after(350, lambda: self._run_prediction(selected))

    def _run_prediction(self, selected: list[str]):
        top_predictions = predict_top_n(self.bundle, set(selected), top_n=3)

        for widget in self.results_container.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.results_container,
            text="TOP POSSIBLE MATCHES",
            font=(FONT_FAMILY, 15, "bold"),
            text_color=self.colors["text"],
        ).pack(anchor="w", pady=(0, 10))

        for disease, confidence in top_predictions:
            row = ctk.CTkFrame(self.results_container, fg_color=self.colors["card"], corner_radius=12)
            row.pack(fill="x", pady=6)

            header = ctk.CTkFrame(row, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(12, 4))

            ctk.CTkLabel(header, text=disease, font=(FONT_FAMILY, 16, "bold"), text_color=self.colors["text"]).pack(
                side="left"
            )
            pct_label = ctk.CTkLabel(header, text="0%", font=(FONT_FAMILY, 15, "bold"), text_color=self.colors["accent"])
            pct_label.pack(side="right")

            bar = ctk.CTkProgressBar(row, height=14, corner_radius=7, progress_color=self.colors["accent"])
            bar.set(0)
            bar.pack(fill="x", padx=16, pady=(0, 14))

            animate_progressbar(bar, confidence / 100, duration_ms=650)
            animate_counter(pct_label, round(confidence), duration_ms=650, suffix="%")

        best_disease, best_confidence = top_predictions[0]
        self.db.add_record(selected, best_disease, best_confidence)

        self.predict_button.configure(state="normal", text="🔍  PREDICT DISEASE")
        self.show_toast(f"Saved prediction: {best_disease}", kind="success")

    # ==================================================================
    # HISTORY PAGE
    # ==================================================================

    def _build_history_page(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=40, pady=(25, 10))

        ctk.CTkLabel(
            header, text="PREDICTION HISTORY", font=(FONT_FAMILY, 26, "bold"), text_color=self.colors["text"]
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="⬇  Export CSV",
            width=150,
            height=36,
            corner_radius=10,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            command=self._export_history,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            header,
            text="🗑  Clear All",
            width=130,
            height=36,
            corner_radius=10,
            fg_color=self.colors["danger"],
            hover_color=self.colors["danger_hover"],
            command=self._clear_history,
        ).pack(side="right")

        self.search_entry = ctk.CTkEntry(
            parent, placeholder_text="🔎  Search by disease or symptom...", height=38, corner_radius=10
        )
        self.search_entry.pack(fill="x", padx=40, pady=(0, 12))
        self.search_entry.bind("<KeyRelease>", lambda _e: self._refresh_history_list())

        self.history_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.history_scroll.pack(fill="both", expand=True, padx=40, pady=(0, 20))

        self._refresh_history_list()

    def _refresh_history_list(self):
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        search_term = self.search_entry.get().strip() if hasattr(self, "search_entry") else ""
        records = self.db.fetch_all(search=search_term or None)

        if not records:
            ctk.CTkLabel(
                self.history_scroll,
                text="No matching prediction history." if search_term else "No prediction history yet.",
                font=(FONT_FAMILY, 14),
                text_color=self.colors["text_muted"],
            ).pack(pady=30)
            return

        for record_id, symptoms, disease, confidence, date_time in records:
            row = ctk.CTkFrame(self.history_scroll, fg_color=self.colors["card"], corner_radius=12)
            row.pack(fill="x", pady=6)

            text_col = ctk.CTkFrame(row, fg_color="transparent")
            text_col.pack(side="left", fill="x", expand=True, padx=16, pady=10)

            ctk.CTkLabel(
                text_col,
                text=f"{disease}   •   {confidence:.1f}% confidence",
                font=(FONT_FAMILY, 14, "bold"),
                text_color=self.colors["text"],
                anchor="w",
            ).pack(fill="x")

            ctk.CTkLabel(
                text_col,
                text=f"Symptoms: {symptoms}",
                font=(FONT_FAMILY, 12),
                text_color=self.colors["text_muted"],
                anchor="w",
            ).pack(fill="x")

            ctk.CTkLabel(
                text_col, text=date_time, font=(FONT_FAMILY, 11), text_color=self.colors["text_muted"], anchor="w"
            ).pack(fill="x")

            ctk.CTkButton(
                row,
                text="🗑",
                width=36,
                height=36,
                corner_radius=8,
                fg_color="transparent",
                text_color=self.colors["danger"],
                hover_color=self.colors["accent_soft"],
                command=lambda rid=record_id: self._delete_history_row(rid),
            ).pack(side="right", padx=12)

    def _delete_history_row(self, record_id: int):
        self.db.delete_record(record_id)
        self._refresh_history_list()
        self.show_toast("Entry deleted", kind="info")

    def _clear_history(self):
        if not messagebox.askyesno("Clear History", "Delete all prediction history? This cannot be undone."):
            return
        self.db.clear_all()
        self._refresh_history_list()
        self.show_toast("History cleared", kind="info")

    def _export_history(self):
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
            initialfile="prediction_history.csv",
            title="Export prediction history",
        )
        if not dest:
            return
        self.db.export_csv(dest)
        self.show_toast("History exported", kind="success")

    # ==================================================================
    # TOAST NOTIFICATIONS
    # ==================================================================

    def show_toast(self, message: str, kind: str = "info", duration_ms: int = 2200):
        color = {
            "success": self.colors["success"],
            "danger": self.colors["danger"],
            "info": self.colors["accent"],
        }.get(kind, self.colors["accent"])

        if self._toast_frame is not None and self._toast_frame.winfo_exists():
            self._toast_frame.destroy()

        toast = ctk.CTkLabel(
            self,
            text=message,
            font=(FONT_FAMILY, 13, "bold"),
            fg_color=color,
            text_color="#0f1115",
            corner_radius=10,
            height=40,
            padx=18,
        )
        toast.place(relx=0.5, rely=1.05, anchor="center")
        self._toast_frame = toast

        def _step(progress):
            toast.place(relx=0.5, rely=1.05 - 0.15 * progress, anchor="center")

        animate(toast, 220, _step, on_done=lambda: self.after(duration_ms, lambda: self._hide_toast(toast)))

    def _hide_toast(self, toast):
        if not toast.winfo_exists():
            return

        def _step(progress):
            toast.place(relx=0.5, rely=0.9 + 0.15 * progress, anchor="center")

        animate(toast, 200, _step, on_done=lambda: toast.destroy() if toast.winfo_exists() else None)

    # ==================================================================
    # SHUTDOWN
    # ==================================================================

    def on_closing(self):
        # HistoryDatabase opens short-lived connections per call, so there
        # is no long-lived connection object to close here -- but this is
        # kept as the single, consistent exit path for both the sidebar
        # "Exit" button and the window's close ("X") button.
        self.destroy()

# ==============================================================================
# ENTRY POINT
# ==============================================================================


def main():
    app = DiseasePredictionApp()
    app.mainloop()


if __name__ == "__main__":
    main()