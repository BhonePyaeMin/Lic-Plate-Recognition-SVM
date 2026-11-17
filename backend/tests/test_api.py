from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from tests.fixtures import encode, iou_xywh, make_scene

ERROR_KEYS = {"success", "message", "error_code"}


def post(client, data: bytes, name: str = "car.jpg", mime: str = "image/jpeg"):
    return client.post("/api/v1/detect", files={"file": (name, data, mime)})


def decode_data_url(url: str) -> np.ndarray:
    header, payload = url.split(",", 1)
    assert header.startswith("data:image/")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == ERROR_KEYS
    assert body["success"] is False
    assert body["error_code"] == code
    assert body["message"] and "Traceback" not in body["message"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["max_upload_mb"] == 10
    assert "jpg" in body["allowed_extensions"]
    # MATLAB must report honestly that it is not available.
    assert body["matlab"]["available"] is False and body["matlab"]["state"] == "disabled"
    assert body["ocr"]["available"] is True


def test_empty_upload(client):
    assert_error(post(client, b""), 400, "EMPTY_FILE")


def test_missing_file_field(client):
    r = client.post("/api/v1/detect")
    assert_error(r, 400, "INVALID_REQUEST")


def test_unsupported_extension(client):
    assert_error(post(client, b"hello world", "notes.txt", "text/plain"), 415, "UNSUPPORTED_FILE_TYPE")


def test_unsupported_mime_with_image_extension(client):
    img, _ = make_scene()
    assert_error(post(client, encode(img), "car.jpg", "text/html"), 415, "UNSUPPORTED_FILE_TYPE")


def test_extension_content_mismatch(client):
    img, _ = make_scene()
    # PNG bytes labelled .jpg: extension and signature disagree.
    assert_error(post(client, encode(img, ".png"), "car.jpg"), 415, "UNSUPPORTED_FILE_TYPE")


def test_corrupted_image(client):
    assert_error(post(client, b"\x00not an image at all" * 20, "car.jpg"), 400, "INVALID_IMAGE")


def test_truncated_image(client):
    img, _ = make_scene()
    data = encode(img)
    r = post(client, data[: len(data) // 3])
    assert r.status_code == 400 and r.json()["success"] is False


def test_oversized_file(make_client):
    client = make_client(max_upload_mb=1)
    big = b"\xff\xd8\xff" + b"0" * (1024 * 1024 + 10)
    assert_error(post(client, big), 413, "FILE_TOO_LARGE")


def test_pixel_limit(make_client):
    client = make_client(max_image_pixels=1000)
    img, _ = make_scene()
    assert_error(post(client, encode(img)), 400, "INVALID_IMAGE")


def test_filename_is_not_a_path(client):
    # Path traversal in the filename must not matter: nothing touches disk.
    img, _ = make_scene()
    r = post(client, encode(img), "../../etc/passwd.jpg")
    assert r.status_code == 200


def test_no_plate_image(client, recognizer):
    img, _ = make_scene(with_plate=False)
    r = post(client, encode(img))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == ERROR_KEYS
    assert body["success"] is False and body["error_code"] == "PLATE_NOT_FOUND"
    assert recognizer.calls == 0  # nothing to read, OCR must not run


def test_blank_image_has_no_plate(client):
    blank = np.full((300, 400, 3), 128, np.uint8)
    assert post(client, encode(blank)).json()["error_code"] == "PLATE_NOT_FOUND"


def test_valid_image_pipeline(client, recognizer):
    img, gt = make_scene(text="XYZ9876")
    recognizer.text = "XYZ 9876"
    r = post(client, encode(img))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"success", "message", "result"} and body["success"] is True
    res = body["result"]
    assert res["detection_status"] == "recognized"
    assert res["plate_text"] == "XYZ9876"
    assert res["raw_ocr_text"] == "XYZ 9876"
    assert res["confidence"] == 0.9
    assert (res["image_width"], res["image_height"]) == (img.shape[1], img.shape[0])
    assert res["processing_time_ms"] > 0
    assert res["matlab_used"] is False
    bb = res["bounding_box"]
    assert iou_xywh(gt, (bb["x"], bb["y"], bb["width"], bb["height"])) > 0.5
    annotated = decode_data_url(res["annotated_image"])
    assert annotated.shape[:2] == img.shape[:2]
    crop = decode_data_url(res["plate_crop"])
    assert crop.shape[0] > 10 and crop.shape[1] > crop.shape[0]


def test_confidence_is_null_when_engine_gives_none(client, recognizer):
    recognizer.confidence = None
    img, _ = make_scene()
    assert post(client, encode(img)).json()["result"]["confidence"] is None


def test_located_but_unreadable(client, recognizer):
    recognizer.text = ""
    img, _ = make_scene()
    body = post(client, encode(img)).json()
    assert body["success"] is True
    res = body["result"]
    assert res["detection_status"] == "located_unreadable"
    assert res["plate_text"] is None and res["confidence"] is None


def test_short_ocr_noise_is_not_a_plate(client, recognizer):
    recognizer.text = "A-1"
    img, _ = make_scene()
    assert post(client, encode(img)).json()["result"]["detection_status"] == "located_unreadable"


def test_png_and_webp_accepted(client):
    img, _ = make_scene()
    assert post(client, encode(img, ".png"), "car.png", "image/png").status_code == 200
    assert post(client, encode(img, ".webp"), "car.webp", "image/webp").status_code == 200


def test_unhandled_error_is_sanitised(make_client, recognizer):
    def boom(_):
        raise RuntimeError("secret internal detail")

    recognizer.recognize = boom
    client = make_client(raise_server_exceptions=False)
    img, _ = make_scene()
    r = post(client, encode(img))
    assert_error(r, 500, "PROCESSING_FAILED")
    assert "secret" not in r.text
