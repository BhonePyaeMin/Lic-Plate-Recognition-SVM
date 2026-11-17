from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.image_preprocessing import deskew, prepare_for_ocr, resize_to_max_side
from app.services.plate_detector import ContourPlateDetector, PlateCandidate
from app.services.plate_recognizer import RapidOcrRecognizer, _reading_order, normalize_plate_text
from tests.fixtures import iou_xywh, make_scene


@pytest.mark.parametrize(
    "raw,expected",
    [("ab 12-cd", "AB12CD"), ("  ", ""), ("O0 Il", "O0IL"), ("AB·12", "AB12")],
)
def test_normalize_is_conservative(raw, expected):
    # No O/0 or I/1 substitution: ambiguity must stay visible.
    assert normalize_plate_text(raw) == expected


@pytest.mark.parametrize("tilt", [0, 6, -9])
def test_detector_finds_plate_on_synthetic_scene(tilt):
    img, gt = make_scene(tilt_deg=tilt)
    cands = ContourPlateDetector().detect(img)
    assert cands, "expected at least one candidate"
    best = cands[0]
    assert iou_xywh(gt, (best.x, best.y, best.width, best.height)) > 0.5
    assert 0.0 <= best.score <= 1.0


def test_detector_candidates_are_ranked_and_in_bounds():
    img, _ = make_scene()
    cands = ContourPlateDetector().detect(img)
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    h, w = img.shape[:2]
    for c in cands:
        assert 0 <= c.x and 0 <= c.y and c.x + c.width <= w and c.y + c.height <= h


def test_detector_maps_back_from_downscaled_image():
    img, gt = make_scene(size=(720, 1080))
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gt2 = tuple(v * 2 for v in gt)
    best = ContourPlateDetector(max_side=960).detect(big)[0]
    assert iou_xywh(gt2, (best.x, best.y, best.width, best.height)) > 0.5


def test_iou():
    a = PlateCandidate(0, 0, 10, 10, 1, "t")
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(PlateCandidate(20, 20, 5, 5, 1, "t")) == 0.0


def test_deskew_reduces_tilt():
    img, _ = make_scene(text="ABC1234", tilt_deg=0, noise=0)
    x, y, w, h = 400, 430, 280, 70
    crop = img[y : y + h, x : x + w]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
    tilted = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def measured_angle(c):
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ys, xs = np.nonzero(b)
        return abs(np.polyfit(xs, ys, 1)[0])  # slope of text mass

    assert measured_angle(deskew(tilted)) < measured_angle(tilted)


def test_deskew_leaves_flat_crop_unchanged():
    img, _ = make_scene(noise=0)
    crop = img[430:500, 400:680]
    assert deskew(crop).shape == crop.shape


def test_prepare_for_ocr_upscales_and_is_three_channel():
    small = np.full((30, 90, 3), 200, np.uint8)
    out = prepare_for_ocr(small)
    assert out.ndim == 3 and out.shape[2] == 3 and out.shape[0] > 96


def test_resize_to_max_side():
    img = np.zeros((2000, 1000, 3), np.uint8)
    out, scale = resize_to_max_side(img, 1000)
    assert out.shape[0] == 1000 and scale == pytest.approx(0.5)
    same, s = resize_to_max_side(out, 1000)
    assert same is out and s == 1.0


def test_reading_order_keeps_single_row_left_to_right():
    box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    result = [(box(120, 10, 200, 60), "C1234", 0.9), (box(10, 14, 100, 64), "AB", 0.9)]
    assert [r[1] for r in _reading_order(result)] == ["AB", "C1234"]
    two_rows = [(box(10, 80, 200, 140), "BOTTOM", 0.9), (box(10, 10, 200, 60), "TOP", 0.9)]
    assert [r[1] for r in _reading_order(two_rows)] == ["TOP", "BOTTOM"]


def test_real_ocr_reads_clean_text_if_engine_present():
    rec = RapidOcrRecognizer()
    ok, _ = rec.is_available()
    if not ok:
        pytest.skip("OCR engine not installed")
    img = np.full((90, 300, 3), 255, np.uint8)
    cv2.putText(img, "ABC1234", (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    out = rec.recognize(img)
    assert normalize_plate_text(out.raw_text) == "ABC1234"
    assert out.confidence is not None and 0 < out.confidence <= 1
