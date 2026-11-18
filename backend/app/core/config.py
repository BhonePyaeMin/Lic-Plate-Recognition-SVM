"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _list(name: str, default: str) -> list[str]:
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = 10
    max_image_pixels: int = 40_000_000  # reject decompression bombs before decoding
    allowed_extensions: tuple[str, ...] = ("jpg", "jpeg", "png", "webp")
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    ocr_language: str = "en"
    ocr_candidates: int = 3  # how many ranked candidate regions are sent to OCR
    detector_max_side: int = 960  # detection runs on a downscaled copy
    enable_matlab: bool = True  # use MATLAB when its Engine is installed; harmless otherwise
    enable_svm: bool = True  # HOG + SVM candidate verification when a trained model file exists
    svm_model_path: str = ""  # empty = backend/models/plate_svm.npz
    svm_min_margin: Optional[float] = None  # None = use the threshold stored with the model
    enable_svm_ocr: bool = True  # character-level SVM OCR engine when models/ocr_svm.npz exists
    svm_ocr_model_path: str = ""  # empty = backend/models/ocr_svm.npz

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    return Settings(
        max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        max_image_pixels=_int("MAX_IMAGE_PIXELS", 40_000_000),
        allowed_extensions=tuple(
            e.lower().lstrip(".") for e in _list("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp")
        ),
        cors_origins=_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"),
        ocr_language=os.environ.get("OCR_LANGUAGE", "en"),
        ocr_candidates=max(1, _int("OCR_CANDIDATES", 3)),
        detector_max_side=max(320, _int("DETECTOR_MAX_SIDE", 960)),
        enable_matlab=_bool("ENABLE_MATLAB", True),
        enable_svm=_bool("ENABLE_SVM", True),
        svm_model_path=os.environ.get("SVM_MODEL_PATH", "").strip(),
        svm_min_margin=_float("SVM_MIN_MARGIN"),
        enable_svm_ocr=_bool("ENABLE_SVM_OCR", True),
        svm_ocr_model_path=os.environ.get("SVM_OCR_MODEL_PATH", "").strip(),
    )
