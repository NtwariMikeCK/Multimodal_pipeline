#!/usr/bin/env python3
"""
auth_gate.py
============
Multimodal Authentication Gate — CLI app
=========================================

Flow
----
1. FACE VERIFICATION
   User chooses:  (a) upload/select an image file   OR
                  (b) capture face from webcam

2. VOICE VERIFICATION  (only if face passes)
   User chooses:  (a) upload/select an audio file   OR
                  (b) record voice from microphone

3. PRODUCT RECOMMENDATION  (only if both pass)
   → Shows the recommendation for a selected record in the merged df

Usage
-----
    python auth_gate.py

Optional flags:
    --image-csv  path/to/image_features.csv   (default: image_features.csv) gotten from multimodal_pipeline.ipynb
    --audio-csv  path/to/audio_features.csv   (default: audio_features.csv) gotten from multimodal_pipeline.ipynb
    --face-thresh  0.80    (cosine similarity threshold for face)
    --voice-thresh 0.80    (cosine similarity threshold for voice)
"""

import argparse
import sys
import os
import time
import textwrap

import numpy as np
from PIL import Image

from face_verifier  import FaceVerifier
from voice_verifier import VoiceVerifier

import joblib
import pandas as pd
import xgboost as xgb


# ─── ANSI colours ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"


# ─── Banner ────────────────────────────────────────────────────────────────────

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║       MULTIMODAL IDENTITY AUTHENTICATION GATE        ║
║         Face Recognition + Voice Verification        ║
╚══════════════════════════════════════════════════════╝{RESET}
"""


# ─── Helper: safe input ────────────────────────────────────────────────────────

def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def choose(question: str, options: list[str]) -> int:
    """Display numbered menu, return 0-based index."""
    print(f"\n{bold(question)}")
    for i, opt in enumerate(options, 1):
        print(f"  {cyan(str(i))}. {opt}")
    while True:
        raw = prompt("→ Choice: ")
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(red(f"  Please enter a number between 1 and {len(options)}."))


# ─── FACE: capture from webcam ─────────────────────────────────────────────────

def capture_face_from_webcam() -> Image.Image | None:
    """Open webcam, show preview, capture on SPACE, quit on Q."""
    try:
        import cv2
    except ImportError:
        print(red("  [!] opencv-python not installed. Run: pip install opencv-python"))
        return None

    print(yellow("  [CAM] Opening webcam…  Press SPACE to capture, Q to cancel."))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(red("  [!] Could not open webcam."))
        return None

    frame_captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Face Capture — SPACE to capture, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame_captured = frame.copy()
            break
        elif key in (ord("q"), ord("Q"), 27):   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_captured is None:
        print(yellow("  [!] Capture cancelled."))
        return None

    # Convert BGR → RGB → PIL
    rgb = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    print(green("  [OK] Face image captured."))
    return pil


# ─── AUDIO: record from microphone ─────────────────────────────────────────────

def record_voice_from_mic(duration: int = 4, sr: int = 22050) -> np.ndarray | None:
    """Record `duration` seconds from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print(red("  [!] sounddevice not installed. Run: pip install sounddevice"))
        return None

    print(yellow(f"  [MIC] Recording {duration} seconds… say 'Yes, approve' or 'Confirm transaction'"))
    print(yellow("        Recording starts NOW ↓"))

    for i in range(3, 0, -1):
        print(f"        {i}…")
        time.sleep(0.8)

    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        y = audio.flatten()
        print(green("  [OK] Recording complete."))
        return y
    except Exception as exc:
        print(red(f"  [!] Recording failed: {exc}"))
        return None


# ─── STEP 1: Face verification ─────────────────────────────────────────────────

def run_face_step(verifier: FaceVerifier) -> str | None:
    """
    Ask user to provide a face image (file or webcam).
    Returns the identified person name, or None if verification fails.
    """
    print(f"\n{bold('━━━  STEP 1 — FACE VERIFICATION  ━━━')}")
    print(f"  Known persons in system: {cyan(', '.join(verifier.known_persons))}")

    choice = choose("How do you want to provide your face?",
                    ["Upload / select an image file",
                     "Capture using webcam"])

    pil_img  = None
    img_path = None

    if choice == 0:   # file upload
        while True:
            img_path = prompt("  Enter path to image file: ")
            if os.path.isfile(img_path):
                break
            print(red(f"  [!] File not found: {img_path}"))

    else:             # webcam
        pil_img = capture_face_from_webcam()
        if pil_img is None:
            print(red("  [ACCESS DENIED] No face image obtained."))
            return None

    print(yellow("  Analysing face…"))
    try:
        if img_path:
            name, score = verifier.identify_from_path(img_path)
        else:
            name, score = verifier.identify_from_pil(pil_img)
    except Exception as exc:
        print(red(f"  [!] Error during face analysis: {exc}"))
        return None

    # Print the bar for similarity
    bar = _similarity_bar(score)
    print(f"  Similarity: {bar}  {score:.4f}")

    # Check compare the similarity with threshold
    if score < verifier.threshold:
        print(red(f"\n  ✗ FACE NOT RECOGNISED — access denied."))
        return None

    print(green(f"\n  ✓ Face recognised as: {bold(name)}  (score={score:.4f})"))
    return name





# ─── STEP 2: Voice verification ────────────────────────────────────────────────

def run_voice_step(verifier: VoiceVerifier, claimed_person: str) -> bool:
    """
    Ask user to provide a voice sample (file or microphone).
    Returns True if voice matches the claimed identity.
    """
    print(f"\n{bold('━━━  STEP 2 — VOICE VERIFICATION  ━━━')}")
    print(f"  Verifying voice for: {cyan(claimed_person)}")
    print(f"  Please say one of: "
      f"Yes, approve  ------------------ "
      f"or "
      f"Confirm transaction ----------------- ")

    choice = choose("How do you want to provide your voice?",
                    ["Upload / select an audio file",
                     "Record using microphone"])

    approved = False
    score    = 0.0

    if choice == 0:   # file
        while True:
            audio_path = prompt("  Enter path to audio file: ")
            if os.path.isfile(audio_path):
                break
            print(red(f"  [!] File not found: {audio_path}"))

        print(yellow("  Analysing voice…"))
        try:
            approved, score = verifier.verify_from_path(audio_path, claimed_person)
        except Exception as exc:
            print(red(f"  [!] Error during audio analysis: {exc}"))
            return False

    else:   # microphone
        duration = 4
        try:
            duration_input = prompt("  Recording duration in seconds [4]: ")
            if duration_input.isdigit():
                duration = int(duration_input)
        except Exception:
            pass

        y = record_voice_from_mic(duration=duration)
        if y is None:
            print(red("  [ACCESS DENIED] No voice sample obtained."))
            return False

        print(yellow("  Analysing voice…"))
        try:
            approved, score = verifier.verify_from_array(y, sr=22050, claimed_person=claimed_person)
        except Exception as exc:
            print(red(f"  [!] Error during audio analysis: {exc}"))
            return False

    bar = _similarity_bar(score)
    print(f"  Similarity: {bar}  {score:.4f}")

    if approved:
        print(green(f"\n  ✓ Voice verified for {bold(claimed_person)}  (score={score:.4f})"))
    else:
        print(red(f"\n  ✗ Voice NOT matched for {claimed_person}  (score={score:.4f}) — access denied."))

    return approved


# ─── STEP 3: Product recommendation ───────────────────────────────────────────

def clean_pipeline(record):
    # Load the columns
    columns = joblib.load('model_columns.pkl')
    # Perform one hot encoding
    df = pd.get_dummies(record)
    # check if the columns match the loaded columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    # Add the missing columns as zero, re-arrange them

    # Remove extra columns (if any)
    df = df[columns]

    # Out put the record ready to be used for prediction
    return df
    


def run_recommendation_step(person: str):


    print(f"\n{bold('━━━  STEP 3 — PRODUCT RECOMMENDATION  ━━━')}")
    print(green(f"\n  ✓ Full authentication passed.  Welcome, {bold(person)}!\n"))
    print(f"  Recommendations for {bold(person)}")

    # Get one record from merged data
    df = pd.read_csv('merged_data.csv')

    record = df.iloc[[3]]

    # Get X = df.drop('product_category') and true_y = df['product_category']
    X = record.drop(columns=['product_category'])

    y = record['product_category']
    y_true = y.values[0]

    # Pass it to the clean_pipeline(function)
    X_clean = clean_pipeline(X)


    # Load Booster
    booster = xgb.Booster()
    booster.load_model('xgboost_model.json')

    # Convert X_clean (pandas DataFrame) to DMatrix
    dtest = xgb.DMatrix(X_clean)

    # Make predictions
    y_pred_proba = booster.predict(dtest)  # probabilities
    y_predicted = y_pred_proba.argmax(axis=1)   # predicted class


    # Map prediction to class name
    pred_class_map = {0:'Books', 1:'Clothing', 2:'Electronics', 3:'Groceries', 4:'Sports'}
    y_pred = pred_class_map.get(y_predicted[0], 'Unknown')
    

    print(yellow("      (Recommended product)"))
    if y_pred != y_true:
        print(red(f"{y_pred} "))
    else:
        print(green(f'{y_pred}'))

    print(yellow(f"      (Correct Product class {y_true})"))


# ─── Similarity visualisation ─────────────────────────────────────────────────

def _similarity_bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, int(score * width)))
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if score >= 0.7 else (YELLOW if score >= 0.5 else RED)
    return f"{color}[{bar}]{RESET}"


# ─── Denied splash ────────────────────────────────────────────────────────────

def print_denied():
    print(f"\n{RED}{BOLD}")
    print("  ╔═══════════════════════════════╗")
    print("  ║                               ║")
    print("  ║      ⛔  ACCESS DENIED        ║")
    print("  ║                               ║")
    print("  ╚═══════════════════════════════╝")
    print(RESET)


def print_granted(person: str):
    print(f"\n{GREEN}{BOLD}")
    print("  ╔═══════════════════════════════════╗")
    print(f"  ║   ✅  ACCESS GRANTED              ║")
    print(f"  ║   User: {person:<26}║")
    print("  ╚═══════════════════════════════════╝")
    print(RESET)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Authentication Gate (Face + Voice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python auth_gate.py
              python auth_gate.py --face-thresh 0.6 --voice-thresh 0.75
              python auth_gate.py --image-csv data/image_features.csv \\
                                  --audio-csv data/audio_features.csv
        """)
    )
    parser.add_argument("--image-csv",    default="image_features.csv")
    parser.add_argument("--audio-csv",    default="audio_features.csv")
    parser.add_argument("--face-thresh",  type=float, default=0.80,
                        help="Cosine similarity threshold for face  (default 0.55)")
    parser.add_argument("--voice-thresh", type=float, default=0.80,
                        help="Cosine similarity threshold for voice (default 0.80)")
    args = parser.parse_args()

    print(BANNER)

    # ── Load models ──────────────────────────────────────────────────────────
    print(bold("Loading verification models…"))
    try:
        face_verifier  = FaceVerifier( csv_path=args.image_csv,
                                       threshold=args.face_thresh)
        voice_verifier = VoiceVerifier(csv_path=args.audio_csv,
                                       threshold=args.voice_thresh)
    except FileNotFoundError as exc:
        print(red(str(exc)))
        sys.exit(1)

    print()

    # ── Gate loop (allow retry for demo purposes) ─────────────────────────────
    while True:
        print(bold("Starting authentication sequence…"))

        # STEP 1 — Face
        person = run_face_step(face_verifier)
        if person is None:
            print_denied()
        else:
            # STEP 2 — Voice
            voice_ok = run_voice_step(voice_verifier, claimed_person=person)
            if not voice_ok:
                print_denied()
            else:
                # STEP 3 — Recommendation (placeholder)
                print_granted(person)
                run_recommendation_step(person)

        # ── Ask to try again ───────────────────────────────────────────────
        again = choose("What would you like to do next?",
                       ["Try again (simulate another attempt)",
                        "Exit"])
        if again == 1:
            print(cyan("\nGoodbye!\n"))
            break
        print()


if __name__ == "__main__":
    main()
