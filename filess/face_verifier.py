"""
face_verifier.py
================
Pure Pretrained Verification (embedding cosine similarity).

No training required. Loads image_features.csv, averages per-person embeddings,
then compares an input face against all known persons.
"""

import os
import csv
import numpy as np
from PIL import Image
from feature_utils import embed_image, embed_image_pil


class FaceVerifier:
    """
    Loads stored embeddings from image_features.csv and identifies
    the closest matching known person via cosine similarity.
    """

    def __init__(self, csv_path: str = "image_features.csv", threshold: float = 0.90):
        """
        Parameters
        ----------
        csv_path  : path to image_features.csv produced by feature_utils.py
        threshold : cosine-similarity cutoff (0-1).  Below this → "Unknown".
                    Tune upward to be stricter, downward to be more lenient.
        """
        self.threshold = threshold
        self.person_embeddings: dict[str, np.ndarray] = {}   # name → mean embedding
        self._load(csv_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[FaceVerifier] '{csv_path}' not found.\n"
                "Run  python feature_utils.py <data_dir>  first."
            )

        buckets: dict[str, list[np.ndarray]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                person = row["person"]
                emb = np.array([float(v) for k, v in row.items()
                                if k not in ("person", "file")], dtype=np.float32)
                buckets.setdefault(person, []).append(emb)

        for person, embs in buckets.items():
            mean_emb = np.mean(embs, axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)
            self.person_embeddings[person] = mean_emb

        print(f"[FaceVerifier] Loaded {len(self.person_embeddings)} known persons: "
              f"{list(self.person_embeddings.keys())}")

    # ── public ───────────────────────────────────────────────────────────────

    def identify_from_path(self, img_path: str) -> tuple[str, float]:
        """
        Identify a person from an image file path.

        Returns
        -------
        (name, similarity)  where name is "Unknown" if below threshold.
        """
        emb = embed_image(img_path)
        return self._compare(emb)

    def identify_from_pil(self, pil_img: Image.Image) -> tuple[str, float]:
        """Identify from a PIL Image (e.g. live camera capture)."""
        emb = embed_image_pil(pil_img)
        return self._compare(emb)

    def _compare(self, emb: np.ndarray) -> tuple[str, float]:
        best_name  = "Unknown"
        best_score = -1.0

        for person, stored_emb in self.person_embeddings.items():
            score = float(np.dot(emb, stored_emb))   # cosine sim (both L2-normed)
            if score > best_score:
                best_score = score
                best_name  = person

        if best_score < self.threshold:
            return "Unknown", best_score
        return best_name, best_score

    @property
    def known_persons(self) -> list[str]:
        return list(self.person_embeddings.keys())
