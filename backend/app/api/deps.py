"""Dependency providers. Tests override these with FastAPI dependency_overrides."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, load_settings
from app.services.matlab_adapter import MatlabAdapter
from app.services.plate_detector import ContourPlateDetector, PlateDetector
from app.services.plate_recognizer import PlateRecognizer, RapidOcrRecognizer
from app.services.svm_ocr import SvmCharRecognizer
from app.services.svm_plate import SvmVerifier


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


DEFAULT_SVM_MODEL = Path(__file__).resolve().parents[2] / "models" / "plate_svm.npz"


@lru_cache
def get_svm() -> SvmVerifier:
    s = get_settings()
    path = Path(s.svm_model_path) if s.svm_model_path else DEFAULT_SVM_MODEL
    return SvmVerifier(path, enabled=s.enable_svm, min_margin=s.svm_min_margin)


DEFAULT_SVM_OCR_MODEL = Path(__file__).resolve().parents[2] / "models" / "ocr_svm.npz"


@lru_cache
def get_svm_ocr() -> SvmCharRecognizer:
    s = get_settings()
    path = Path(s.svm_ocr_model_path) if s.svm_ocr_model_path else DEFAULT_SVM_OCR_MODEL
    return SvmCharRecognizer(path, enabled=s.enable_svm_ocr)
