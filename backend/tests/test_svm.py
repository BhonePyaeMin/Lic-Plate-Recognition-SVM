"""SVM verifier: features, model loading states, detector wrapper and API modes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.api.deps import DEFAULT_SVM_MODEL
from app.services.plate_detector import ContourPlateDetector, PlateCandidate
from app.services.svm_plate import FEATURE_DIM, SvmPlateDetector, SvmVerifier, extract_features, squash
from tests.fixtures import encode, make_scene


def write_model(path: Path, bias: float = 0.0, weight_fill: float = 0.0, meta: dict | None = None) -> Path:
    np.savez(
        path,
        mean=np.zeros(FEATURE_DIM, np.float32),
        scale=np.ones(FEATURE_DIM, np.float32),
        coef=np.full(FEATURE_DIM, weight_fill, np.float32),
        intercept=np.float32(bias),
    )
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return path


class StubVerifier:
    """Verifier whose margin depends on the crop's mean brightness (no model file)."""

    ready = True
    min_margin = 0.0

    def __init__(self, threshold: float = 128.0):
        self.threshold = threshold

    def status(self):
        return {"available": True, "enabled": True, "state": "ready", "detail": "stub"}

    def margin(self, crop: np.ndarray) -> float:
        return float(crop.mean() - self.threshold) / 50.0


# ----------------------------------------------------------------- features
def test_feature_vector_shape_and_determinism():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (40, 150), dtype=np.uint8)
    a, b = extract_features(patch), extract_features(patch)
    assert a.shape == (FEATURE_DIM,) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_features_accept_color_and_any_size():
    bgr = np.random.default_rng(1).integers(0, 255, (25, 90, 3), dtype=np.uint8)
    assert extract_features(bgr).shape == (FEATURE_DIM,)
    with pytest.raises(ValueError):
        extract_features(np.zeros((0, 10), np.uint8))


def test_features_ignore_plate_polarity():
    """Unsigned gradients: dark-on-light and light-on-dark give the same descriptor."""
    patch = np.random.default_rng(2).integers(0, 255, (32, 128), dtype=np.uint8)
    assert np.abs(extract_features(patch) - extract_features(255 - patch)).max() < 1e-4


def test_squash_is_monotonic_and_bounded():
    values = [squash(m) for m in (-50, -1, 0, 1, 50)]
    assert values == sorted(values) and 0.0 <= values[0] and values[-1] <= 1.0
    assert squash(0.0) == pytest.approx(0.5)


# ----------------------------------------------------------- model loading
def test_verifier_disabled():
    v = SvmVerifier.disabled()
    assert not v.ready and v.status()["state"] == "disabled" and v.status()["available"] is False


def test_verifier_missing_model(tmp_path):
    v = SvmVerifier(tmp_path / "nope.npz")
    assert not v.ready and v.status()["state"] == "no_model"
    with pytest.raises(RuntimeError):
        v.margin(np.zeros((10, 40), np.uint8))


def test_verifier_corrupt_model_does_not_crash(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a model")
    v = SvmVerifier(bad)
    assert not v.ready and v.status()["state"] == "failed"


def test_verifier_rejects_wrong_feature_size(tmp_path):
    path = tmp_path / "small.npz"
    np.savez(path, mean=np.zeros(10), scale=np.ones(10), coef=np.zeros(10), intercept=np.float32(0))
    assert SvmVerifier(path).status()["state"] == "failed"


def test_verifier_margin_is_linear(tmp_path):
    v = SvmVerifier(write_model(tmp_path / "m.npz", bias=1.5, weight_fill=0.0))
    assert v.ready and v.margin(np.full((20, 80), 100, np.uint8)) == pytest.approx(1.5)


def test_min_margin_comes_from_model_metadata_unless_overridden(tmp_path):
    path = write_model(tmp_path / "m.npz", meta={"suggested_min_margin": -0.4, "trained_on": "test data"})
    assert SvmVerifier(path).min_margin == pytest.approx(-0.4)
    assert SvmVerifier(path, min_margin=0.25).min_margin == pytest.approx(0.25)
    assert "test data" in SvmVerifier(path).status()["detail"]


# ------------------------------------------------------- detector wrapper
class FixedDetector:
    name = "fixed"

    def __init__(self, cands):
        self.cands = cands

    def detect(self, image):
        return list(self.cands)


def two_candidates():
    img = np.zeros((200, 400, 3), np.uint8)
    img[20:60, 20:160] = 220     # bright region: positive margin with the stub
    img[120:160, 200:340] = 30   # dark region: negative margin
    dark = PlateCandidate(200, 120, 140, 40, 0.9, "a")   # heuristic prefers the dark region
    bright = PlateCandidate(20, 20, 140, 40, 0.5, "b")
    return img, [dark, bright]


def test_wrapper_reranks_without_rejecting():
    img, cands = two_candidates()
    det = SvmPlateDetector(FixedDetector(cands), StubVerifier(), reject=False)
    out = det.detect(img)
    assert len(out) == 2
    assert out[0].source == "b" and out[0].svm_score > 0 > out[1].svm_score  # SVM overturns the heuristic order
    assert all(0.0 <= c.score <= 1.0 for c in out)
    assert det.name.endswith("rerank")


def test_wrapper_rejects_below_threshold():
    img, cands = two_candidates()
    out = SvmPlateDetector(FixedDetector(cands), StubVerifier(), reject=True).detect(img)
    assert [c.source for c in out] == ["b"]


def test_wrapper_skips_empty_crops():
    img, _ = two_candidates()
    outside = PlateCandidate(900, 900, 50, 20, 0.8, "x")
    assert SvmPlateDetector(FixedDetector([outside]), StubVerifier()).detect(img) == []


# -------------------------------------------------------------------- API
def post(client, data, **form):
    return client.post("/api/v1/detect", files={"file": ("car.jpg", data, "image/jpeg")}, data=form)


def test_health_reports_svm_state(make_client):
    body = make_client().get("/health").json()
    assert body["svm"]["state"] == "disabled" and body["svm"]["available"] is False
    body = make_client(svm=StubVerifier()).get("/health").json()
    assert body["svm"]["state"] == "ready" and body["svm"]["available"] is True


def test_invalid_detector_mode(client):
    img, _ = make_scene()
    r = post(client, encode(img), detector_mode="yolo")
    assert r.status_code == 400 and r.json()["error_code"] == "INVALID_PARAMS"


def test_svm_mode_without_model_is_503(client):
    img, _ = make_scene()
    r = post(client, encode(img), detector_mode="svm")
    assert r.status_code == 503 and r.json()["error_code"] == "SVM_UNAVAILABLE"


def test_auto_and_contour_modes_work_without_model(client):
    img, _ = make_scene()
    for mode in ("auto", "contour"):
        body = post(client, encode(img), detector_mode=mode).json()
        assert body["success"] is True
        assert body["result"]["svm_used"] is False and body["result"]["svm_score"] is None
        assert body["result"]["detector"] == ContourPlateDetector.name


def test_svm_mode_reports_score(make_client):
    client = make_client(svm=StubVerifier(threshold=0.0))  # every crop has margin >= 0
    img, _ = make_scene()
    body = post(client, encode(img), detector_mode="svm").json()
    r = body["result"]
    assert body["success"] is True and r["svm_used"] is True
    assert isinstance(r["svm_score"], float)
    assert r["detector"].endswith("+hog-svm")


def test_auto_mode_uses_rerank_when_model_ready(make_client):
    client = make_client(svm=StubVerifier(threshold=1e9))  # would reject everything in svm mode
    img, _ = make_scene()
    body = post(client, encode(img)).json()  # default mode is auto: re-rank only, never reject
    assert body["success"] is True and body["result"]["svm_used"] is True
    assert body["result"]["detector"].endswith("hog-svm-rerank")
    strict = post(client, encode(img), detector_mode="svm").json()
    assert strict["success"] is False and strict["error_code"] == "PLATE_NOT_FOUND"


def test_filters_catalog_lists_detector_modes(client):
    assert client.get("/api/v1/filters").json()["detector_modes"] == ["auto", "contour", "svm"]


# ------------------------------------------------- shipped model (if present)
@pytest.mark.skipif(not DEFAULT_SVM_MODEL.exists(), reason="no trained model in backend/models")
def test_shipped_model_prefers_plate_over_clutter():
    verifier = SvmVerifier(DEFAULT_SVM_MODEL)
    assert verifier.ready, verifier.status()
    img, (x, y, w, h) = make_scene(text="KLM5821", seed=3)
    plate_margin = verifier.margin(img[max(0, y - 6) : y + h + 6, max(0, x - 10) : x + w + 10])
    # A plate-shaped crop of road and sky has no characters.
    background_margin = verifier.margin(img[5:55, 5:215])
    assert plate_margin > background_margin
    assert plate_margin > 0 > background_margin
