from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_matlab, get_recognizer, get_settings, get_svm, get_svm_ocr
from app.core.config import load_settings
from app.main import create_app
from app.services.matlab_adapter import MatlabAdapter
from app.services.plate_recognizer import OcrOutput
from app.services.svm_ocr import SvmCharRecognizer
from app.services.svm_plate import SvmVerifier


class FakeRecognizer:
    """Deterministic OCR stand-in so API tests never depend on model files."""

    name = "fake-ocr"
    language = "en"

    def __init__(self, text: str = "ABC 1234", confidence: float | None = 0.9):
        self.text, self.confidence, self.calls = text, confidence, 0

    def is_available(self):
        return True, "fake"

    def recognize(self, image: np.ndarray) -> OcrOutput:
        self.calls += 1
        return OcrOutput(raw_text=self.text, confidence=self.confidence if self.text else None)


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def make_client(recognizer):
    def _make(raise_server_exceptions: bool = True, matlab=None, svm=None, svm_ocr=None, **setting_overrides) -> TestClient:
        app = create_app()
        settings = dataclasses.replace(load_settings(), **setting_overrides)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_recognizer] = lambda: recognizer
        adapter = matlab if matlab is not None else MatlabAdapter(enabled=False)
        app.dependency_overrides[get_matlab] = lambda: adapter
        verifier = svm if svm is not None else SvmVerifier.disabled()  # keeps API tests independent of the model file
        app.dependency_overrides[get_svm] = lambda: verifier
        char_ocr = svm_ocr if svm_ocr is not None else SvmCharRecognizer.disabled()
        app.dependency_overrides[get_svm_ocr] = lambda: char_ocr
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
