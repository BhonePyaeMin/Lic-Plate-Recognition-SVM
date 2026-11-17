"""License plate localisation.

`PlateDetector` is the interface; `ContourPlateDetector` is the explainable
OpenCV baseline. A learned detector (for example YOLO) can be added by
implementing the same two members (`name`, `detect`) and swapping it in
`app/api/deps.py`; nothing else in the pipeline needs to change.

Baseline method (no training, no model files)
---------------------------------------------
1. Downscale (longest side <= DETECTOR_MAX_SIDE), gray, bilateral denoise.
2. Black-hat (dark characters on a light plate) and top-hat (light characters
   on a dark plate) with a wide rectangular kernel keep only features that are
   smaller than the kernel, i.e. character strokes.
3. A horizontal Sobel gradient of that response is dense where characters sit
   side by side. A wide closing merges the characters into one blob.
4. Otsu threshold, open/erode/dilate to remove specks, external contours.
5. Geometric filter: aspect ratio, area fraction and minimum size.
6. Rank survivors with a weighted score (see `_score`), apply non-maximum
   suppression, pad slightly, and rescale to original image coordinates.

Known limitations
-----------------
* Single-row plates only (aspect ratio 1.8 to 8). Two-row plates are missed.
* Heavy motion blur, strong perspective, glare, or very low resolution break
  the character-density assumption.
* Cluttered scenes with text-like texture (fences, grilles, signage) produce
  false candidates. The pipeline mitigates this by running OCR on the top
  candidates, but a false region can still be returned as "possible plate".
* The score is a heuristic ranking value, not a probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from app.services.image_preprocessing import denoise, resize_to_max_side, to_gray


@dataclass(frozen=True)
class PlateCandidate:
    x: int
    y: int
    width: int
    height: int
    score: float  # 0..1 heuristic, higher is better. Not a probability.
    source: str  # which pass produced it

    def iou(self, other: "PlateCandidate") -> float:
        ix = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        iy = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        inter = ix * iy
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union else 0.0


class PlateDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[PlateCandidate]:
        """Return candidates ranked best first, in the coordinates of `image`."""
        ...


class ContourPlateDetector:
    name = "opencv-contour-baseline"

    # Geometry limits. Target aspect ~4 covers most single-row plates.
    MIN_ASPECT = 1.8
    MAX_ASPECT = 8.0
    TARGET_ASPECT = 4.0
    MIN_AREA_FRAC = 0.0008
    MAX_AREA_FRAC = 0.25
    MIN_SCORE = 0.30
    MAX_CANDIDATES = 8

    def __init__(self, max_side: int = 960):
        self.max_side = max_side

    # ------------------------------------------------------------------ API
    def detect(self, image: np.ndarray) -> list[PlateCandidate]:
        small, scale = resize_to_max_side(image, self.max_side)
        gray = denoise(to_gray(small))
        h, w = gray.shape[:2]

        found: list[PlateCandidate] = []
        for mode in ("dark_on_light", "light_on_dark"):
            mask = self._candidate_mask(gray, mode)
            found.extend(self._candidates_from_mask(mask, gray, mode))

        ranked = self._non_max_suppression(sorted(found, key=lambda c: c.score, reverse=True))
        ranked = [c for c in ranked if c.score >= self.MIN_SCORE][: self.MAX_CANDIDATES]

        # The mask hugs the character blob, but a plate also has a margin
        # around its characters. Pad by typical margin ratios (12 percent of
        # width, 25 percent of height), then map back to the original
        # resolution and clip to the image.
        full_h, full_w = image.shape[:2]
        out: list[PlateCandidate] = []
        for c in ranked:
            pad_x, pad_y = int(c.width * 0.12), int(c.height * 0.25)
            x0, y0 = max(0, c.x - pad_x), max(0, c.y - pad_y)
            x1, y1 = min(w, c.x + c.width + pad_x), min(h, c.y + c.height + pad_y)
            out.append(
                PlateCandidate(
                    x=int(round(x0 / scale)),
                    y=int(round(y0 / scale)),
                    width=min(full_w, int(round((x1 - x0) / scale))),
                    height=min(full_h, int(round((y1 - y0) / scale))),
                    score=c.score,
                    source=c.source,
                )
            )
        return out

    # ------------------------------------------------------------- internals
    def _candidate_mask(self, gray: np.ndarray, mode: str) -> np.ndarray:
        h, w = gray.shape[:2]
        # Kernel scales with image width so it matches roughly one character.
        kw = max(9, int(round(w / 75)) | 1)
        kh = max(3, int(round(kw * 0.4)) | 1)
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        op = cv2.MORPH_BLACKHAT if mode == "dark_on_light" else cv2.MORPH_TOPHAT
        response = cv2.morphologyEx(gray, op, rect)

        grad = np.abs(cv2.Sobel(response, cv2.CV_32F, 1, 0, ksize=-1))  # Scharr
        if grad.max() <= 0:
            return np.zeros_like(gray)
        grad = (255 * (grad - grad.min()) / (grad.max() - grad.min())).astype(np.uint8)
        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        # Close horizontally so neighbouring characters fuse into one blob.
        wide = cv2.getStructuringElement(cv2.MORPH_RECT, (kw * 2 + 1, kh))
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, wide)
        _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Remove thin noise, then restore blob size.
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, kh + 2)))
        return mask

    def _candidates_from_mask(self, mask: np.ndarray, gray: np.ndarray, source: str) -> list[PlateCandidate]:
        h, w = gray.shape[:2]
        image_area = float(h * w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = cv2.Canny(gray, 80, 200)
        result: list[PlateCandidate] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0 or cw < 40 or ch < 12:
                continue
            aspect = cw / ch
            frac = cw * ch / image_area
            if not (self.MIN_ASPECT <= aspect <= self.MAX_ASPECT):
                continue
            if not (self.MIN_AREA_FRAC <= frac <= self.MAX_AREA_FRAC):
                continue
            score = self._score(gray[y : y + ch, x : x + cw], edges[y : y + ch, x : x + cw], aspect, cv2.contourArea(contour) / (cw * ch))
            result.append(PlateCandidate(x, y, cw, ch, score, source))
        return result

    def _score(self, roi_gray: np.ndarray, roi_edges: np.ndarray, aspect: float, fill: float) -> float:
        """Weighted score in [0, 1]. Criteria (weights in brackets):

        * aspect  (0.30): Gaussian closeness to TARGET_ASPECT.
        * edges   (0.30): Canny edge density, saturating at 18 percent. Text
                          is edge-dense, flat paint and sky are not.
        * contrast(0.20): gray standard deviation, saturating at 55. A plate
                          has high local contrast between characters and base.
        * fill    (0.20): contour area / bounding box area. Plate blobs are
                          solid rectangles, scattered texture is not.
        """
        aspect_s = math.exp(-(((aspect - self.TARGET_ASPECT) / 1.8) ** 2))
        edge_s = min(float(roi_edges.mean()) / 255.0 / 0.18, 1.0)
        contrast_s = min(float(roi_gray.std()) / 55.0, 1.0)
        fill_s = min(max(fill, 0.0), 1.0)
        return round(0.30 * aspect_s + 0.30 * edge_s + 0.20 * contrast_s + 0.20 * fill_s, 4)

    @staticmethod
    def _non_max_suppression(cands: list[PlateCandidate], iou_thr: float = 0.35) -> list[PlateCandidate]:
        kept: list[PlateCandidate] = []
        for c in cands:
            if all(c.iou(k) < iou_thr for k in kept):
                kept.append(c)
        return kept
