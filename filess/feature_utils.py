"""
feature_utils.py
================
Shared utilities for extracting and saving face + audio features.
Run this once to build image_features.csv and audio_features.csv
from your data/ folder before running the auth system.
"""

import os
import csv
import numpy as np

# ── Image deps ──────────────────────────────────────────────────────────────
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()  # register HEIC/HEIF support with PIL


import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


# ── Audio deps ───────────────────────────────────────────────────────────────
import librosa


# ────────────────────────────────────────────────────────────────────────────
# IMAGE  (ResNet-18 pretrained embeddings, 512-dim) Using Tensorflow
# ────────────────────────────────────────────────────────────────────────────
# Using tensorflow
# import tensorflow as tf

# def _build_image_model_tf():
#     base_model = tf.keras.applications.ResNet50(
#         weights="imagenet",
#         include_top=False,
#         pooling="avg"   # gives embedding directly
#     )
#     return base_model


# ────────────────────────────────────────────────────────────────────────────
# IMAGE  (ResNet-18 pretrained embeddings, 512-dim)
# ────────────────────────────────────────────────────────────────────────────

def _build_image_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()          # strip classifier → 512-dim
    model.eval()
    return model, weights.transforms()

_IMG_MODEL, _IMG_TRANSFORM = _build_image_model()


def embed_image(img_path: str) -> np.ndarray:
    """Return a 512-dim L2-normalised embedding for one image file."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Skipping {img_path}: {e}")
        return None
    tensor = _IMG_TRANSFORM(img).unsqueeze(0)   # (1, C, H, W)
    with torch.no_grad():
        emb = _IMG_MODEL(tensor).squeeze().numpy()
    emb /= (np.linalg.norm(emb) + 1e-9)
    return emb


def embed_image_pil(pil_img: Image.Image) -> np.ndarray:
    """Embed a PIL Image directly (used by the auth gate for live capture)."""
    tensor = _IMG_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = _IMG_MODEL(tensor).squeeze().numpy()
    emb /= (np.linalg.norm(emb) + 1e-9)
    return emb


# ────────────────────────────────────────────────────────────────────────────
# AUDIO  (40 MFCCs + spectral roll-off + RMS energy → 42-dim)
# ────────────────────────────────────────────────────────────────────────────

def embed_audio(audio_path: str, sr: int = 22050) -> np.ndarray:
    """Return a 42-dim L2-normalised audio feature vector."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)   # (40,)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()        # scalar
    energy  = librosa.feature.rms(y=y).mean()                            # scalar
    feat = np.append(mfcc, [rolloff, energy]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat


def embed_audio_array(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Embed a raw numpy audio array (used by the auth gate for live recording)."""
    mfcc    = librosa.feature.mfcc(y=y.astype(float), sr=sr, n_mfcc=40).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y.astype(float), sr=sr).mean()
    energy  = librosa.feature.rms(y=y.astype(float)).mean()
    feat = np.append(mfcc, [rolloff, energy]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat


# ────────────────────────────────────────────────────────────────────────────
# CSV builders  (run once)
# ────────────────────────────────────────────────────────────────────────────

def build_image_features_csv(data_dir: str, out_csv: str = "image_features.csv"):
    """Walk data/images/<person>/*.jpg and write embeddings to CSV."""
    images_dir = os.path.join(data_dir, "images")
    rows = []
    for person in sorted(os.listdir(images_dir)):
        person_dir = os.path.join(images_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(person_dir, fname)
            emb = embed_image(fpath)
            rows.append({"person": person, "file": fname, **{f"e{i}": v for i, v in enumerate(emb)}})

    if not rows:
        print("[WARN] No images found – image_features.csv not written.")
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Saved {len(rows)} image embeddings → {out_csv}")


def build_audio_features_csv(data_dir: str, out_csv: str = "audio_features.csv"):
    """Walk data/audio/<person>/*.m4a and write features to CSV."""
    audio_dir = os.path.join(data_dir, "audio")
    rows = []
    for person in sorted(os.listdir(audio_dir)):
        person_dir = os.path.join(audio_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".m4a", ".wav", ".mp3", ".ogg", ".flac")):
                continue
            fpath = os.path.join(person_dir, fname)
            feat = embed_audio(fpath)
            rows.append({"person": person, "file": fname, **{f"f{i}": v for i, v in enumerate(feat)}})

    if not rows:
        print("[WARN] No audio files found – audio_features.csv not written.")
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Saved {len(rows)} audio feature vectors → {out_csv}")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_image_features_csv(data_dir)
    build_audio_features_csv(data_dir)
