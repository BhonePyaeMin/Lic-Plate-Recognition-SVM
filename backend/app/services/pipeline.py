"""End-to-end plate pipeline: detect -> crop -> correct -> OCR -> annotate.

Each step is a small function so it can be tested alone.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.schemas.detection import BoundingBox, DetectionResult, PreprocessingInfo
from app.services.image_preprocessing import deskew, prepare_for_ocr
from app.services.plate_detector import PlateCandidate, PlateDetector
from app.services.plate_recognizer import OcrOutput, PlateRecognizer, normalize_plate_text

MIN_PLATE_CHARS = 4  # shorter strings are treated as OCR noise, not a plate
MAX_PREVIEW_SIDE = 1600


@dataclass
class _Attempt:
    candidate: PlateCandidate
    crop: np.ndarray
    ocr: OcrOutput
    text: str

    @property
    def plausible(self) -> bool:
        return len(self.text) >= MIN_PLATE_CHARS

    @property
    def rank_value(self) -> float:
        # Blend geometry score with OCR confidence (0.5 if the engine gave none).
        conf = self.ocr.confidence if self.ocr.confidence is not None else 0.5
        return 0.5 * self.candidate.score + 0.5 * conf


def crop_region(image: np.ndarray, c: PlateCandidate) -> np.ndarray:
    return image[c.y : c.y + c.height, c.x : c.x + c.width]


def annotate(image: np.ndarray, c: PlateCandidate, label: Optional[str]) -> np.ndarray:
    out = image.copy()
    thickness = max(2, int(round(max(out.shape[:2]) / 400)))
    cv2.rectangle(out, (c.x, c.y), (c.x + c.width, c.y + c.height), (60, 180, 40), thickness)
    if label:
        scale = max(0.6, thickness / 3)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y0 = max(c.y - th - base - 6, 0)
        cv2.rectangle(out, (c.x, y0), (c.x + tw + 10, y0 + th + base + 6), (60, 180, 40), -1)
        cv2.putText(out, label, (c.x + 5, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def to_data_url(image: np.ndarray, fmt: str) -> str:
    h, w = image.shape[:2]
    if max(h, w) > MAX_PREVIEW_SIDE:
        s = MAX_PREVIEW_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, image, params)
    if not ok:
        raise RuntimeError("image encoding failed")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def run_pipeline(
    image: np.ndarray,
    detector: PlateDetector,
    recognizer: PlateRecognizer,
    max_ocr_candidates: int = 3,
    preprocessing: Optional[PreprocessingInfo] = None,
) -> Optional[DetectionResult]:
    """Return a DetectionResult, or None when no candidate region is found."""
    started = time.perf_counter()
    height, width = image.shape[:2]

    user_cropped = preprocessing is not None and preprocessing.crop is not None
    candidates = detector.detect(image)
    if not candidates and not user_cropped:
        return None

    attempts: list[_Attempt] = []
    for cand in candidates[:max_ocr_candidates]:
        crop = deskew(crop_region(image, cand))
        if crop.size == 0:
            continue
        ocr = recognizer.recognize(prepare_for_ocr(crop))
        attempts.append(_Attempt(cand, crop, ocr, normalize_plate_text(ocr.raw_text)))

    if user_cropped and not any(a.plausible for a in attempts):
        # The user drew a crop, often tight around the plate. The baseline
        # detector assumes the plate is a small part of the frame, so as a
        # fallback read the whole cropped area. Only used when it yields text.
        whole = PlateCandidate(0, 0, width, height, 0.0, "user-crop")
        crop = deskew(image)
        ocr = recognizer.recognize(prepare_for_ocr(crop))
        attempt = _Attempt(whole, crop, ocr, normalize_plate_text(ocr.raw_text))
        if attempt.plausible:
            attempts.insert(0, attempt)
    if not attempts:
        return None

    readable = [a for a in attempts if a.plausible]
    best = max(readable, key=lambda a: a.rank_value) if readable else attempts[0]
    status = "recognized" if readable else "located_unreadable"

    annotated = annotate(image, best.candidate, best.text if readable else None)
    result = DetectionResult(
        detection_status=status,
        plate_text=best.text if readable else None,
        raw_ocr_text=best.ocr.raw_text or None,
        confidence=best.ocr.confidence if readable else None,
        bounding_box=BoundingBox(x=best.candidate.x, y=best.candidate.y, width=best.candidate.width, height=best.candidate.height),
        detector="user-crop (whole selected area)" if best.candidate.source == "user-crop" else detector.name,
        detector_score=best.candidate.score,
        ocr_engine=recognizer.name,
        image_width=width,
        image_height=height,
        processing_time_ms=0.0,
        annotated_image=to_data_url(annotated, ".jpg"),
        plate_crop=to_data_url(best.crop, ".png"),
    )
    if preprocessing is not None and (preprocessing.crop or preprocessing.steps):
        result.preprocessing = preprocessing
        result.matlab_used = any(s.engine == "matlab" for s in preprocessing.steps)
        result.input_image = to_data_url(image, ".jpg")
    result.processing_time_ms = round((time.perf_counter() - started) * 1000.0, 1)
    return result
