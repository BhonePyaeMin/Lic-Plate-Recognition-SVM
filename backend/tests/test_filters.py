from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidParamsError, MatlabUnavailableError
from app.services.filters import CATALOG, resolve_params
from app.services.matlab_adapter import MatlabAdapter
from app.services.processing import Step, parse_crop, parse_steps, run_steps
from tests.fixtures import encode, make_scene

IMG, _ = make_scene(size=(240, 360), plate_xywh=(130, 140, 110, 30), noise=4)
REF, _ = make_scene(size=(240, 360), plate_xywh=(130, 140, 110, 30), noise=10, seed=3)


class FakeMatlab(MatlabAdapter):
    """Pretends MATLAB is running so engine routing can be tested without MATLAB."""

    def __init__(self, fail=False, passthrough=False):
        super().__init__(enabled=True)
        self.fail = fail
        self.passthrough = passthrough

    @property
    def ready(self):
        return True

    def run_operation(self, op_id, params, image, ref=None):
        if self.fail:
            raise MatlabUnavailableError("boom")
        return image if self.passthrough else np.full_like(image, 7)


@pytest.mark.parametrize("op_id", list(CATALOG))
def test_every_operation_runs_with_defaults(op_id):
    """All catalog operations must be usable with their default parameters."""
    op = CATALOG[op_id]
    params = resolve_params(op, None)
    out = op.run(IMG.copy(), params, REF if op.needs_reference else None)
    assert out.dtype == np.uint8 and out.ndim in (2, 3) and out.size > 0
    if op.output == "mask":
        assert set(np.unique(out)) <= {0, 255}


def test_catalog_covers_course_modules():
    ids = set(CATALOG)
    assert {"hist_equalization", "hist_matching", "brightness_contrast", "mean", "median", "laplacian", "highpass",
            "fourier_low", "fourier_high", "fourier_band"} <= ids
    assert {"sobel", "prewitt", "canny", "otsu", "region_growing", "rgb_seg", "hsv_seg", "dilate", "erode", "open",
            "close", "plate_shapes"} <= ids
    assert {"translate", "scale", "rotate", "shear", "radial", "register"} <= ids


def test_operations_actually_change_the_image():
    for op_id in ("hist_equalization", "brightness_contrast", "mean", "median", "laplacian", "fourier_high", "scale", "rotate"):
        op = CATALOG[op_id]
        out = op.run(IMG.copy(), resolve_params(op, None), None)
        assert out.shape != IMG.shape or not np.array_equal(out, IMG), op_id


def test_known_results():
    flat = np.full((20, 20, 3), 100, np.uint8)
    assert CATALOG["brightness_contrast"].run(flat, {"contrast": 1.0, "brightness": 30}, None)[0, 0, 0] == 130
    assert CATALOG["scale"].run(flat, {"factor": 2.0, "interpolation": "nearest"}, None).shape == (40, 40, 3)
    t = np.zeros((30, 30, 3), np.uint8)
    t[10, 10] = 255
    moved = CATALOG["translate"].run(t, {"tx": 5, "ty": 3, "interpolation": "nearest", "mapping": "reverse"}, None)
    assert moved[13, 15, 0] == 255 and np.count_nonzero(moved[:, :, 0]) == 1
    sq = np.dstack([np.tile(np.arange(0, 200, 4, dtype=np.uint8), (50, 1))] * 3)
    rot = CATALOG["rotate"].run(sq, {"angle": 90, "enlarge": "no", "interpolation": "nearest", "mapping": "reverse"}, None)
    assert np.abs(rot.astype(int) - np.rot90(sq).astype(int)).max() <= 4


def test_forward_mapping_leaves_holes_but_reverse_does_not():
    img = np.full((60, 60, 3), 200, np.uint8)
    p = {"angle": 20, "enlarge": "no", "interpolation": "nearest"}
    fwd = CATALOG["rotate"].run(img, {**p, "mapping": "forward"}, None)
    rev = CATALOG["rotate"].run(img, {**p, "mapping": "reverse"}, None)
    centre = (slice(20, 40), slice(20, 40))
    assert (rev[centre] == 200).all()
    assert (fwd[centre] == 0).any()


def test_histogram_matching_moves_toward_reference():
    dark = np.full((50, 50, 3), 40, np.uint8) + np.random.default_rng(0).integers(0, 20, (50, 50, 3), dtype=np.uint8)
    bright = (dark.astype(int) + 150).clip(0, 255).astype(np.uint8)
    out = CATALOG["hist_matching"].run(dark, {}, bright)
    assert abs(out.mean() - bright.mean()) < abs(dark.mean() - bright.mean())


def test_fourier_low_pass_smooths_and_band_validates():
    noise = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    out = CATALOG["fourier_low"].run(noise, resolve_params(CATALOG["fourier_low"], {"cutoff": 6}), None)
    assert out.std() < noise.std()
    with pytest.raises(InvalidParamsError):
        resolve_params(CATALOG["fourier_band"], {"inner": 50, "outer": 20})


def test_hsv_hue_wraparound():
    red = np.zeros((10, 10, 3), np.uint8)
    red[:] = (0, 0, 255)
    p = resolve_params(CATALOG["hsv_seg"], {"h_min": 340, "h_max": 20, "s_min": 0.5, "s_max": 1, "v_min": 0.5, "v_max": 1})
    assert CATALOG["hsv_seg"].run(red, p, None).all()


def test_region_growing_fills_flat_rectangle():
    g = np.full((60, 80, 3), 50, np.uint8)
    g[20:41, 20:61] = 180
    params = resolve_params(CATALOG["region_growing"], {"seed_x": 50, "seed_y": 50, "tolerance": 10})
    out = CATALOG["region_growing"].run(g, params, None)
    assert np.array_equal(out > 0, g[:, :, 0] == 180)


def test_morphology_opening_removes_isolated_pixel():
    bw = np.zeros((20, 20), np.uint8)
    bw[10, 10] = 255
    out = CATALOG["open"].run(bw, resolve_params(CATALOG["open"], {"size": 3}), None)
    assert not out.any()


def test_registration_aligns_shifted_copy():
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 1.2)
    shifted = np.roll(base, (12, 18), axis=(0, 1))
    out = CATALOG["register"].run(shifted, resolve_params(CATALOG["register"], None), base)
    err_before = np.abs(shifted[40:200, 60:260].astype(int) - base[40:200, 60:260]).mean()
    err_after = np.abs(out[40:200, 60:260].astype(int) - base[40:200, 60:260]).mean()
    assert err_after < err_before


def test_register_and_match_need_reference():
    for op_id in ("register", "hist_matching"):
        with pytest.raises(InvalidParamsError):
            CATALOG[op_id].run(IMG, resolve_params(CATALOG[op_id], None), None)


@pytest.mark.parametrize("bad", [{"kernel": 0}, {"kernel": 999}, {"kernel": "x"}, {"kernel": float("nan")}])
def test_param_validation(bad):
    with pytest.raises(InvalidParamsError):
        resolve_params(CATALOG["mean"], bad)


def test_select_validation():
    with pytest.raises(InvalidParamsError):
        resolve_params(CATALOG["scale"], {"interpolation": "lanczos"})


def test_output_size_is_capped():
    with pytest.raises(InvalidParamsError):
        CATALOG["scale"].run(np.zeros((3000, 3000, 3), np.uint8), {"factor": 4.0, "interpolation": "nearest"}, None)


def test_parse_steps_and_crop():
    assert parse_steps(None) == [] and parse_steps("") == []
    assert parse_steps(json.dumps([{"id": "mean", "params": {"kernel": 3}}]))[0].params == {"kernel": 3}
    for raw in ("not json", json.dumps([{"id": "nope"}]), json.dumps([{"id": "mean"}] * 9), json.dumps({"id": "mean"})):
        with pytest.raises(InvalidParamsError):
            parse_steps(raw)
    assert parse_crop(json.dumps({"x": -5, "y": 10, "width": 100, "height": 50}), 200, 100) == (0, 10, 95, 50)
    with pytest.raises(InvalidParamsError):
        parse_crop(json.dumps({"x": 0, "y": 0, "width": 5, "height": 5}), 200, 100)
    with pytest.raises(InvalidParamsError):
        parse_crop(json.dumps({"x": 500, "y": 0, "width": 50, "height": 50}), 200, 100)


def test_engine_selection_and_fallback():
    off = MatlabAdapter(enabled=False)
    steps = [Step("mean", {"kernel": 3})]
    _, rec = run_steps(IMG, steps, "auto", off)
    assert rec[0]["engine"] == "opencv"
    with pytest.raises(MatlabUnavailableError):
        run_steps(IMG, steps, "matlab", off)

    out, rec = run_steps(IMG, steps, "auto", FakeMatlab())
    assert rec[0]["engine"] == "matlab" and (out == 7).all()
    out, rec = run_steps(IMG, steps, "auto", FakeMatlab(fail=True))
    assert rec[0]["engine"] == "opencv" and rec[0]["fallback"] is True
    with pytest.raises(MatlabUnavailableError):
        run_steps(IMG, steps, "matlab", FakeMatlab(fail=True))
    # OpenCV-only operations are never sent to MATLAB in auto mode
    _, rec = run_steps(IMG, [Step("register", {"interpolation": "bilinear"})], "auto", FakeMatlab(), REF)
    assert rec[0]["engine"] == "opencv"


def _fake_matlab_modules(monkeypatch, failures: int):
    """Install a fake `matlab.engine` whose start_matlab fails `failures` times, then works."""
    import sys
    import types

    calls = {"n": 0}

    class Engine:
        def genpath(self, p):
            return p

        def addpath(self, *a, **k):
            pass

    def start_matlab():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise RuntimeError("engine did not start")
        return Engine()

    pkg, eng = types.ModuleType("matlab"), types.ModuleType("matlab.engine")
    eng.start_matlab = start_matlab
    pkg.engine = eng
    monkeypatch.setitem(sys.modules, "matlab", pkg)
    monkeypatch.setitem(sys.modules, "matlab.engine", eng)
    return calls


def test_matlab_start_retries_after_a_transient_failure(monkeypatch):
    calls = _fake_matlab_modules(monkeypatch, failures=2)
    adapter = MatlabAdapter(enabled=True, start_attempts=3, retry_delay=0)
    adapter._state = "starting"
    adapter._start()
    assert adapter.ready and calls["n"] == 3 and adapter.status().state == "ready"


def test_matlab_start_reports_failure_after_all_attempts(monkeypatch):
    calls = _fake_matlab_modules(monkeypatch, failures=99)
    adapter = MatlabAdapter(enabled=True, start_attempts=3, retry_delay=0)
    adapter._state = "starting"
    adapter._start()
    st = adapter.status()
    assert not adapter.ready and calls["n"] == 3 and st.state == "failed" and "RuntimeError" in st.detail


def test_matlab_adapter_status_without_package():
    st = MatlabAdapter(enabled=True).status()  # matlabengine is not installed in the test environment
    assert st.available is False and st.state in {"no_package", "starting", "failed"}
    assert MatlabAdapter(enabled=False).status().state == "disabled"


# ------------------------------------------------------------------ API
def _post(client, path, img, **fields):
    return client.post(path, files={"file": ("car.jpg", encode(img), "image/jpeg")}, data=fields)


def test_filters_catalog_endpoint(client):
    body = client.get("/api/v1/filters").json()
    assert body["success"] and {o["module"] for o in body["operations"]} == {1, 2, 3}
    assert all("params" in o and "label" in o for o in body["operations"])


def test_apply_endpoint_returns_processed_image(client):
    steps = json.dumps([{"id": "scale", "params": {"factor": 2, "interpolation": "bicubic"}}])
    r = _post(client, "/api/v1/filters/apply", IMG, steps=steps)
    assert r.status_code == 200
    res = r.json()["result"]
    assert (res["width"], res["height"]) == (720, 480)
    assert res["preprocessing"]["steps"][0]["engine"] == "opencv"


def test_apply_mask_output_is_flagged(client):
    r = _post(client, "/api/v1/filters/apply", IMG, steps=json.dumps([{"id": "sobel"}]))
    assert r.json()["result"]["image"].startswith("data:image/png")
    assert r.json()["result"]["preprocessing"]["steps"][0]["output"] == "mask"


def test_apply_with_crop_and_reference(client):
    r = client.post(
        "/api/v1/filters/apply",
        files={"file": ("a.jpg", encode(IMG), "image/jpeg"), "reference": ("b.jpg", encode(REF), "image/jpeg")},
        data={"crop": json.dumps({"x": 100, "y": 100, "width": 150, "height": 100}), "steps": json.dumps([{"id": "hist_matching"}])},
    )
    assert r.status_code == 200
    res = r.json()["result"]
    assert (res["width"], res["height"]) == (150, 100) and res["preprocessing"]["crop"]["x"] == 100


@pytest.mark.parametrize(
    "fields,status,code",
    [
        ({"steps": "oops"}, 400, "INVALID_PARAMS"),
        ({"steps": json.dumps([{"id": "mean", "params": {"kernel": 500}}])}, 400, "INVALID_PARAMS"),
        ({"steps": json.dumps([{"id": "hist_matching"}])}, 400, "INVALID_PARAMS"),
        ({"engine": "matlab", "steps": json.dumps([{"id": "mean"}])}, 503, "MATLAB_UNAVAILABLE"),
        ({"engine": "gpu"}, 400, "INVALID_PARAMS"),
        ({"crop": json.dumps({"x": 0, "y": 0, "width": 2, "height": 2})}, 400, "INVALID_PARAMS"),
    ],
)
def test_apply_errors(client, fields, status, code):
    r = _post(client, "/api/v1/filters/apply", IMG, **fields)
    assert r.status_code == status
    assert r.json() == {"success": False, "message": r.json()["message"], "error_code": code}


def test_detect_uses_crop_and_filters(client):
    scene, _ = make_scene(size=(720, 1080))
    crop = {"x": 300, "y": 350, "width": 500, "height": 250}
    r = _post(client, "/api/v1/detect", scene, crop=json.dumps(crop), steps=json.dumps([{"id": "median", "params": {"kernel": 3}}]))
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert (res["image_width"], res["image_height"]) == (500, 250)  # coordinates are in cropped space
    assert res["preprocessing"]["crop"] == crop
    assert res["preprocessing"]["steps"][0]["id"] == "median"
    assert res["input_image"].startswith("data:image/jpeg")
    assert res["matlab_used"] is False


def test_tight_user_crop_falls_back_to_whole_area(client, recognizer):
    """A crop that fills the frame with the plate is read as a whole when the detector finds no text."""
    plate_only = np.full((90, 330, 3), 245, np.uint8)
    cv2.putText(plate_only, "ABC1234", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (20, 20, 20), 4)
    recognizer.text = "ABC 1234"
    crop = {"x": 0, "y": 0, "width": 330, "height": 90}
    body = _post(client, "/api/v1/detect", plate_only, crop=json.dumps(crop)).json()
    assert body["success"] is True
    res = body["result"]
    assert res["plate_text"] == "ABC1234"
    assert res["detector"].startswith("user-crop") or res["detector"] == "opencv-contour-baseline"


def test_detect_without_preprocessing_has_no_input_image(client):
    scene, _ = make_scene()
    res = _post(client, "/api/v1/detect", scene).json()["result"]
    assert res["preprocessing"] is None and res["input_image"] is None


def test_detect_reports_matlab_used(make_client):
    client = make_client(matlab=FakeMatlab(passthrough=True))
    scene, _ = make_scene()
    res = _post(client, "/api/v1/detect", scene, steps=json.dumps([{"id": "mean"}])).json()["result"]
    assert res["matlab_used"] is True and res["preprocessing"]["steps"][0]["engine"] == "matlab"
