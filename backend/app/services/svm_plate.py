"""HOG + linear SVM plate verification.

What it does
------------
The OpenCV baseline (`ContourPlateDetector`) proposes candidate regions and
ranks them with a hand-weighted heuristic. This module adds a *learned*
second opinion: every candidate crop is resized to 128x32, described with
Histogram of Oriented Gradients (HOG) features, and scored by a linear
Support Vector Machine trained to tell "licence plate" from "not a plate".

    margin = w . ((hog - mean) / scale) + b

A positive margin means the SVM puts the crop on the plate side of its
decision boundary; larger means further from the boundary. The margin is a
distance-like value, NOT a probability.

`SvmPlateDetector` wraps any `PlateDetector`: it scores each proposed
candidate and re-ranks them by a blend of the heuristic score and the squashed
SVM margin. With `reject=True` it also drops candidates below `min_margin`.

Model file
----------
`models/plate_svm.npz` holds `mean`, `scale`, `coef`, `intercept` as plain
arrays (no pickle), so serving needs only numpy and OpenCV. It is produced by
`backend/scripts/train_svm.py` (needs scikit-learn, see requirements-train.txt).

Limitation (read before trusting it)
------------------------------------
The shipped model is trained only on synthetic, drawn scenes. Its accuracy on
real photographs has not been measured. Retrain on real labelled images with
`scripts/train_svm.py` before relying on it outside demonstrations.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.services.plate_detector import PlateCandidate, PlateDetector

WINDOW = (128, 32)  # (width, height) every crop is resized to; aspect 4:1 like a plate
CELL = 8            # pixels per HOG cell side
BINS = 9            # unsigned orientation bins over 0..180 degrees
FEATURE_DIM = (WINDOW[0] // CELL - 1) * (WINDOW[1] // CELL - 1) * 4 * BINS  # 2x2-cell blocks, stride 1 cell -> 1620


def hog(gray: np.ndarray, cell: int = CELL, bins: int = BINS) -> np.ndarray:
    """Histogram of Oriented Gradients written in numpy (about 0.2 ms for 128x32).

    Written out instead of cv2.HOGDescriptor because OpenCV 5 dropped that class
    and the model must produce identical features on every OpenCV version.
    The image height and width must be multiples of `cell`. Steps: centred
    [-1 0 1] gradients, unsigned orientation, magnitude-weighted votes shared
    between the two nearest of `bins` bins per cell, then 2x2-cell blocks
    (stride one cell) normalised with L2-Hys (clip at 0.2, renormalise).
    Shared by the plate verifier (128x32) and the character classifier (24x48).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    if h % cell or w % cell:
        raise ValueError(f"image {w}x{h} is not a multiple of the {cell}px cell")
    cells_y, cells_x = h // cell, w // cell
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gx[:, 0], gx[:, -1] = g[:, 1] - g[:, 0], g[:, -1] - g[:, -2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    gy[0, :], gy[-1, :] = g[1, :] - g[0, :], g[-1, :] - g[-2, :]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0

    pos = ang / (180.0 / bins) - 0.5            # bin centres at 10, 30, ... degrees for 9 bins
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    lo %= bins
    hi = (lo + 1) % bins

    rows, cols = np.indices(g.shape)
    cell_id = (rows // cell) * cells_x + (cols // cell)
    n = cells_x * cells_y * bins
    hist = np.bincount((cell_id * bins + lo).ravel(), (mag * (1 - frac)).ravel(), minlength=n)
    hist += np.bincount((cell_id * bins + hi).ravel(), (mag * frac).ravel(), minlength=n)
    hist = hist.reshape(cells_y, cells_x, bins)

    nby, nbx = cells_y - 1, cells_x - 1
    blocks = np.stack([hist[i : i + nby, j : j + nbx] for i in (0, 1) for j in (0, 1)], axis=2)
    blocks = blocks.reshape(nby, nbx, 4 * bins)
    blocks = blocks / np.sqrt((blocks ** 2).sum(axis=2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(axis=2, keepdims=True) + 1e-6)
    return blocks.reshape(-1).astype(np.float32)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """HOG descriptor of a crop (BGR or gray, any size). Returns float32 of length FEATURE_DIM."""
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if patch.size == 0:
        raise ValueError("empty patch")
    resized = cv2.resize(patch, WINDOW, interpolation=cv2.INTER_AREA)
    # Unsigned gradients (0..180 degrees) make the descriptor insensitive to
    # dark-on-light versus light-on-dark plates.
    return hog(resized)


@dataclass(frozen=True)
class SvmModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    meta: dict

    def margin(self, features: np.ndarray) -> float:
        z = (features - self.mean) / self.scale
        return float(np.dot(z, self.coef) + self.intercept)


def load_model(path: Path) -> SvmModel:
    with np.load(path, allow_pickle=False) as data:
        mean, scale, coef = data["mean"], data["scale"], data["coef"]
        intercept = float(np.asarray(data["intercept"]).reshape(-1)[0])
    if not (mean.shape == scale.shape == coef.shape == (FEATURE_DIM,)):
        raise ValueError(f"model has {coef.shape} weights but the feature extractor produces {FEATURE_DIM}")
    scale = np.where(scale == 0, 1.0, scale)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return SvmModel(mean.astype(np.float32), scale.astype(np.float32), coef.astype(np.float32), intercept, meta)


class SvmVerifier:
    """Loads the model once and reports honest availability, like MatlabAdapter."""

    def __init__(self, model_path: Optional[Path], enabled: bool = True, min_margin: Optional[float] = None):
        self.enabled = enabled
        self.min_margin = 0.0 if min_margin is None else min_margin
        self.model_path = model_path
        self._model: Optional[SvmModel] = None
        self._state, self._detail = "disabled", "SVM verification is turned off (ENABLE_SVM=false)."
        if not enabled:
            return
        if model_path is None or not Path(model_path).exists():
            self._state = "no_model"
            self._detail = "No trained model found. Run backend/scripts/train_svm.py to create models/plate_svm.npz."
            return
        try:
            self._model = load_model(Path(model_path))
        except Exception as exc:  # corrupt or mismatched file: keep the API running
            self._state, self._detail = "failed", f"Could not load the SVM model ({type(exc).__name__})."
            return
        if min_margin is None:  # use the rejection threshold chosen at training time
            self.min_margin = float(self._model.meta.get("suggested_min_margin", 0.0))
        self._state = "ready"
        trained_on = self._model.meta.get("trained_on", "unknown data")
        self._detail = f"HOG + linear SVM, trained on {trained_on}."

    @classmethod
    def disabled(cls) -> "SvmVerifier":
        return cls(None, enabled=False)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {"available": self.ready, "enabled": self.enabled, "state": self._state, "detail": self._detail}

    def margin(self, crop: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("SVM model is not loaded")
        return self._model.margin(extract_features(crop))


def squash(margin: float) -> float:
    """Map a margin to 0..1 for blending with the heuristic. Not a probability."""
    return 1.0 / (1.0 + math.exp(-2.0 * max(-30.0, min(30.0, margin))))


class SvmPlateDetector:
    """Wrap a candidate proposer; keep and rank candidates by the SVM."""

    HEURISTIC_WEIGHT = 0.4  # remainder goes to the squashed SVM margin

    def __init__(self, base: PlateDetector, verifier: SvmVerifier, reject: bool = True):
        # reject=False re-ranks only and never drops a candidate (the safe choice
        # while the model has no real-photo validation); reject=True also drops
        # candidates whose margin is below verifier.min_margin.
        self.base = base
        self.verifier = verifier
        self.reject = reject
        self.name = f"{base.name}+hog-svm" if reject else f"{base.name}+hog-svm-rerank"

    def detect(self, image: np.ndarray) -> list[PlateCandidate]:
        scored: list[PlateCandidate] = []
        for c in self.base.detect(image):
            crop = image[c.y : c.y + c.height, c.x : c.x + c.width]
            if crop.size == 0:
                continue
            m = self.verifier.margin(crop)
            if self.reject and m < self.verifier.min_margin:
                continue
            blended = self.HEURISTIC_WEIGHT * c.score + (1 - self.HEURISTIC_WEIGHT) * squash(m)
            scored.append(
                PlateCandidate(c.x, c.y, c.width, c.height, round(blended, 4), c.source, svm_score=round(m, 4))
            )
        return sorted(scored, key=lambda c: c.score, reverse=True)
