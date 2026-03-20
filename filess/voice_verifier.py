"""
voice_verifier.py
=================
Pure Pretrained Verification (embedding cosine similarity).

Loads audio_features.csv, averages per-person audio feature vectors,
then compares an input voice sample against all known persons.
"""

import os
import csv
import numpy as np
from feature_utils import embed_audio, embed_audio_array


class VoiceVerifier:
    """
    Loads stored audio features from audio_features.csv and verifies
    whether a given voice sample matches the claimed identity.
    """

    def __init__(self, csv_path: str = "audio_features.csv", threshold: float = 0.80):
        """
        Parameters
        ----------
        csv_path  : path to audio_features.csv produced by feature_utils.py
        threshold : cosine-similarity cutoff (0-1).  Below this → rejected.
                    Audio embeddings are generally tighter, so threshold is higher.
        """
        self.threshold = threshold
        self.person_embeddings: dict[str, np.ndarray] = {}
        self._load(csv_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[VoiceVerifier] '{csv_path}' not found.\n"
                "Run  python feature_utils.py <data_dir>  first."
            )

        buckets: dict[str, list[np.ndarray]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                person = row["person"]
                feat = np.array([float(v) for k, v in row.items()
                                 if k not in ("person", "file")], dtype=np.float32)
                buckets.setdefault(person, []).append(feat)

        for person, feats in buckets.items():
            mean_feat = np.mean(feats, axis=0)
            mean_feat /= (np.linalg.norm(mean_feat) + 1e-9)
            self.person_embeddings[person] = mean_feat

        print(f"[VoiceVerifier] Loaded {len(self.person_embeddings)} known persons: "
              f"{list(self.person_embeddings.keys())}")

    # ── public ───────────────────────────────────────────────────────────────

    def verify_from_path(self, audio_path: str, claimed_person: str) -> tuple[bool, float]:
        """
        Verify audio file against claimed identity.

        Returns
        -------
        (approved, similarity_score)
        """
        feat = embed_audio(audio_path)
        return self._compare(feat, claimed_person)

    def verify_from_array(self, y: np.ndarray, sr: int, claimed_person: str) -> tuple[bool, float]:
        """Verify a raw numpy audio array (live microphone recording)."""
        feat = embed_audio_array(y, sr)
        return self._compare(feat, claimed_person)

    def identify_from_path(self, audio_path: str) -> tuple[str, float]:
        """Identify the most likely speaker regardless of claimed identity."""
        feat = embed_audio(audio_path)
        return self._identify(feat)

    def identify_from_array(self, y: np.ndarray, sr: int) -> tuple[str, float]:
        feat = embed_audio_array(y, sr)
        return self._identify(feat)

    # ── private helpers ──────────────────────────────────────────────────────

    def _compare(self, feat: np.ndarray, claimed_person: str) -> tuple[bool, float]:
        if claimed_person not in self.person_embeddings:
            print(f"[VoiceVerifier] '{claimed_person}' not in enrolled persons.")
            return False, 0.0
        stored = self.person_embeddings[claimed_person]
        score  = float(np.dot(feat, stored))
        return (score >= self.threshold), score

    def _identify(self, feat: np.ndarray) -> tuple[str, float]:
        best_name  = "Unknown"
        best_score = -1.0
        for person, stored in self.person_embeddings.items():
            score = float(np.dot(feat, stored))
            if score > best_score:
                best_score = score
                best_name  = person
        if best_score < self.threshold:
            return "Unknown", best_score
        return best_name, best_score

    @property
    def known_persons(self) -> list[str]:
        return list(self.person_embeddings.keys())
