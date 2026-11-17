"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    )
