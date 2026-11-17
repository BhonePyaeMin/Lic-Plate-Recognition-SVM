"""Dependency providers. Tests override these with FastAPI dependency_overrides."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, load_settings
from app.services.matlab_adapter import MatlabAdapter
from app.services.plate_detector import ContourPlateDetector, PlateDetector
from app.services.plate_recognizer import PlateRecognizer, RapidOcrRecognizer


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_detector() -> PlateDetector:
    return ContourPlateDetector(max_side=get_settings().detector_max_side)


@lru_cache
def get_recognizer() -> PlateRecognizer:
    return RapidOcrRecognizer(language=get_settings().ocr_language)


@lru_cache
def get_matlab() -> MatlabAdapter:
    return MatlabAdapter(enabled=get_settings().enable_matlab)
