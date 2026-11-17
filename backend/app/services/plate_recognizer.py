"""OCR behind a small interface.

The first implementation wraps RapidOCR (ONNX Runtime). It was chosen because
it installs with `pip` alone and ships its model files inside the wheel, so no
system binary (Tesseract) or first-run model download (EasyOCR) is needed.
Replace it by implementing `PlateRecognizer`.

Language: the bundled RapidOCR recognition model covers Chinese and English
text. `OCR_LANGUAGE` is reported to clients for transparency; plate output is
always restricted to A-Z and 0-9 by `normalize_plate_text`.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from app.core.errors import OcrUnavailableError


@dataclass(frozen=True)
class OcrOutput:
    raw_text: str  # engine output, lines joined by a space, unmodified
    confidence: Optional[float]  # mean line score 0..1, None if the engine gave none


class PlateRecognizer(Protocol):
    name: str
    language: str

    def is_available(self) -> tuple[bool, str]: ...

    def recognize(self, image: np.ndarray) -> OcrOutput: ...


def normalize_plate_text(raw: str) -> str:
    """Uppercase and keep only A-Z and 0-9.

    Deliberately conservative: no O/0 or I/1 substitution, no country format
    repair. Guessing characters would hide uncertainty from the user.
    """
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _reading_order(result: list) -> list:
    """Order OCR boxes into rows (top to bottom), each row left to right.

    Sorting by raw top-y alone misorders a single-row plate whose pieces are
    detected a few pixels apart vertically, so boxes whose vertical centres
    differ by less than half a box height are treated as one row.
    """
    def geom(r):
        ys = [p[1] for p in r[0]]
        xs = [p[0] for p in r[0]]
        return (min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs)

    items = sorted(result, key=lambda r: geom(r)[0])
    rows: list[list] = []
    for r in items:
        cy, hgt, _ = geom(r)
        if rows:
            ref_cy, ref_h, _ = geom(rows[-1][0])
            if abs(cy - ref_cy) < 0.5 * min(hgt, ref_h):
                rows[-1].append(r)
                continue
        rows.append([r])
    return [r for row in rows for r in sorted(row, key=lambda r: geom(r)[2])]


class RapidOcrRecognizer:
    name = "rapidocr-onnxruntime"

    def __init__(self, language: str = "en"):
        self.language = language
        self._engine = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()  # lazy init and inference are not guaranteed thread-safe

    def _load(self):
        if self._engine is not None or self._error is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except Exception as exc:  # missing package or broken runtime
            self._error = f"{type(exc).__name__}: {exc}"

    def is_available(self) -> tuple[bool, str]:
        with self._lock:
            self._load()
        if self._engine is None:
            return False, "OCR engine could not be loaded."
        return True, f"{self.name} (language setting: {self.language})"

    def recognize(self, image: np.ndarray) -> OcrOutput:
        with self._lock:
            self._load()
            if self._engine is None:
                raise OcrUnavailableError(
                    "The text recognition engine is not available on the server."
                )
            result, _ = self._engine(image)
        if not result:
            return OcrOutput(raw_text="", confidence=None)
        lines = _reading_order(result)
        texts = [str(r[1]) for r in lines]
        scores = []
        for r in lines:
            try:
                scores.append(float(r[2]))
            except (TypeError, ValueError):
                pass
        confidence = round(sum(scores) / len(scores), 3) if scores else None
        return OcrOutput(raw_text=" ".join(texts), confidence=confidence)
