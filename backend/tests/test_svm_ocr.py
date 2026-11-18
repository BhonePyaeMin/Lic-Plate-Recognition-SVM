"""SVM character recognition: segmentation, model loading, recognizer and API selection."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.api.deps import DEFAULT_SVM_OCR_MODEL
from app.services.image_preprocessing import prepare_for_ocr
from app.services.plate_recognizer import OcrOutput
from app.services.svm_ocr import (
    ALPHABET, GLYPH_FEATURE_DIM, FallbackRecognizer, SvmCharRecognizer, glyph_features, normalize_glyph, render_views,
    segment_characters,
)
from tests.fixtures import encode, make_scene


def draw_plate(text: str, dark_on_light: bool = True, width: int = 300, height: int = 80) -> np.ndarray:
    """A clean drawn plate crop with a margin, like a detector crop."""
    bg, fg = ((240, 240, 240), (20, 20, 20)) if dark_on_light else ((25, 25, 25), (235, 235, 235))
    img = np.full((height + 40, width + 60, 3), 140, np.uint8)
    cv2.rectangle(img, (30, 20), (30 + width, 20 + height), bg, -1)
    cv2.rectangle(img, (30, 20), (30 + width, 20 + height), fg, 2)
    scale = height / 70 * 1.55
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
    cv2.putText(img, text, (30 + (width - tw) // 2, 20 + (height + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, fg, 3, cv2.LINE_AA)
    return img


def write_model(path, bias_class: str = "A") -> None:
    """A model whose weights are zero: every glyph is the class with the largest intercept."""
    intercept = np.zeros(len(ALPHABET), np.float32)
    intercept[ALPHABET.index(bias_class)] = 1.0
    np.savez(
        path,
        classes=np.array([ALPHABET]),
        mean=np.zeros(GLYPH_FEATURE_DIM, np.float32),
        scale=np.ones(GLYPH_FEATURE_DIM, np.float32),
        coef=np.zeros((len(ALPHABET), GLYPH_FEATURE_DIM), np.float32),
        intercept=intercept,
    )


# ------------------------------------------------------------ segmentation
@pytest.mark.parametrize("text", ["KLM5821", "ABC1234", "7JW4309"])
def test_segmentation_finds_each_character_in_order(text):
    glyphs = segment_characters(prepare_for_ocr(draw_plate(text)))
    assert len(glyphs) == len(text)
    xs = [g.box[0] for g in glyphs]
    assert xs == sorted(xs)


def test_segmentation_handles_light_on_dark_and_ignores_spaces():
    assert len(segment_characters(prepare_for_ocr(draw_plate("KLM5821", dark_on_light=False)))) == 7
    assert len(segment_characters(prepare_for_ocr(draw_plate("AB 12 CD")))) == 6


def test_segmentation_returns_nothing_for_blank_or_noise():
    assert segment_characters(np.full((90, 300, 3), 128, np.uint8)) == []
    noise = np.random.default_rng(0).integers(0, 255, (90, 300, 3), dtype=np.uint8)
    assert len(segment_characters(noise)) < 3  # not mistaken for a plate


def test_glyph_features_shape_and_aspect_kept():
    glyphs = segment_characters(prepare_for_ocr(draw_plate("I8W")))  # narrow, round and wide
    assert len(glyphs) == 3
    assert all(glyph_features(g).shape == (GLYPH_FEATURE_DIM,) for g in glyphs)
    assert glyphs[0].aspect < glyphs[2].aspect  # I is narrower than W
    assert normalize_glyph(np.full((10, 40), 255, np.uint8)).shape == (48, 24)


# ----------------------------------------------------------- model loading
def test_recognizer_disabled_and_missing(tmp_path):
    assert SvmCharRecognizer.disabled().status()["state"] == "disabled"
    r = SvmCharRecognizer(tmp_path / "nope.npz")
    assert not r.ready and r.status()["state"] == "no_model" and r.is_available()[0] is False
    with pytest.raises(RuntimeError):
        r.classify(segment_characters(prepare_for_ocr(draw_plate("ABC1234")))[0])


def test_recognizer_corrupt_or_mismatched_model(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"junk")
    assert SvmCharRecognizer(bad).status()["state"] == "failed"
    small = tmp_path / "small.npz"
    np.savez(small, classes=np.array(["AB"]), mean=np.zeros(5), scale=np.ones(5), coef=np.zeros((2, 5)), intercept=np.zeros(2))
    assert SvmCharRecognizer(small).status()["state"] == "failed"


def test_recognize_reports_text_and_per_character_detail(tmp_path):
    write_model(tmp_path / "m.npz", bias_class="K")
    r = SvmCharRecognizer(tmp_path / "m.npz")
    assert r.ready and r.name == "svm-hog-chars"
    out = r.recognize(prepare_for_ocr(draw_plate("ABC1234")))
    assert out.raw_text == "KKKKKKK"            # the stub model answers K for everything
    assert out.confidence is None               # a margin is not a probability, so none is reported
    assert len(out.detail) == 7 and set(out.detail[0]) == {"char", "margin", "x", "y", "width", "height"}
    assert out.detail[0]["margin"] == pytest.approx(1.0)


def test_recognize_without_characters_gives_empty_text(tmp_path):
    write_model(tmp_path / "m.npz")
    out = SvmCharRecognizer(tmp_path / "m.npz").recognize(np.full((90, 300, 3), 128, np.uint8))
    assert out.raw_text == "" and out.detail is None


# --------------------------------------------------------------------- API
class StubSvmOcr:
    name = "svm-hog-chars"
    language = "en"
    ready = True

    def status(self):
        return {"available": True, "enabled": True, "state": "ready", "detail": "stub"}

    def is_available(self):
        return True, "stub"

    def recognize(self, image):
        detail = [{"char": c, "margin": 1.5, "x": i * 10, "y": 0, "width": 8, "height": 20} for i, c in enumerate("ABC1234")]
        return OcrOutput(raw_text="ABC1234", confidence=None, detail=detail, engine=self.name)


def post(client, data, **form):
    return client.post("/api/v1/detect", files={"file": ("car.jpg", data, "image/jpeg")}, data=form)


def test_health_and_catalog_report_svm_ocr(make_client):
    assert make_client().get("/health").json()["svm_ocr"]["state"] == "disabled"
    assert make_client(svm_ocr=StubSvmOcr()).get("/health").json()["svm_ocr"]["available"] is True
    assert make_client().get("/api/v1/filters").json()["ocr_engines"] == ["auto", "rapidocr", "svm"]


def test_invalid_ocr_engine(client):
    img, _ = make_scene()
    r = post(client, encode(img), ocr_engine="tesseract")
    assert r.status_code == 400 and r.json()["error_code"] == "INVALID_PARAMS"


def test_svm_ocr_unavailable_is_503(client):
    img, _ = make_scene()
    r = post(client, encode(img), ocr_engine="svm")
    assert r.status_code == 503 and r.json()["error_code"] == "SVM_OCR_UNAVAILABLE"


def test_default_engine_falls_back_to_rapidocr_when_no_svm_model(client, recognizer):
    img, _ = make_scene()
    body = post(client, encode(img)).json()
    assert body["success"] is True and body["result"]["ocr_engine"] == "fake-ocr"
    assert body["result"]["svm_chars"] is None and body["result"]["ocr_fallback"] is False


def test_svm_engine_returns_text_and_characters(make_client, recognizer):
    client = make_client(svm_ocr=StubSvmOcr())
    img, _ = make_scene()
    body = post(client, encode(img), ocr_engine="svm").json()
    r = body["result"]
    assert body["success"] is True and r["ocr_engine"] == "svm-hog-chars"
    assert r["plate_text"] == "ABC1234" and r["confidence"] is None
    assert [c["char"] for c in r["svm_chars"]] == list("ABC1234")
    assert recognizer.calls == 0  # RapidOCR was not used


def test_render_views_draws_boxes_and_glyph_strip(tmp_path):
    img = prepare_for_ocr(draw_plate("KLM5821"))
    glyphs = segment_characters(img)
    views = render_views(img, glyphs, list("KLM5821"))
    assert views["segmented"].shape == img.shape and views["segmented"].shape[2] == 3
    assert (views["segmented"] != img).any()                     # boxes were drawn
    assert views["glyphs"].shape[0] > 96 and views["glyphs"].shape[1] >= 7 * 48
    assert views["glyphs"].mean() > 200                          # ink black on a white strip


def test_recognize_attaches_views(tmp_path):
    write_model(tmp_path / "m.npz", bias_class="K")
    out = SvmCharRecognizer(tmp_path / "m.npz").recognize(prepare_for_ocr(draw_plate("ABC1234")))
    assert out.engine == "svm-hog-chars" and set(out.views) == {"segmented", "glyphs"}


class ShortSvmOcr(StubSvmOcr):
    """An SVM that finds only two characters, which is too few to be a plate."""

    def recognize(self, image):
        return OcrOutput(raw_text="AB", confidence=None, detail=[], engine=self.name, views={})


def test_fallback_recognizer_keeps_svm_when_it_can_read(recognizer):
    fb = FallbackRecognizer(StubSvmOcr(), recognizer)
    out = fb.recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC1234" and out.engine == "svm-hog-chars" and out.fallback is False
    assert recognizer.calls == 1  # one cross-check call
    assert out.alternative == {"engine": "fake-ocr", "text": "ABC1234", "confidence": 0.9}


def test_cross_check_never_breaks_a_good_svm_reading():
    class Boom:
        name = "boom"

        def is_available(self):
            return True, "x"

        def recognize(self, image):
            raise RuntimeError("second engine crashed")

    out = FallbackRecognizer(StubSvmOcr(), Boom()).recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC1234" and out.alternative is None and out.fallback is False


def test_cross_check_reports_a_disagreement(recognizer):
    recognizer.text = "ABC 1284"
    out = FallbackRecognizer(StubSvmOcr(), recognizer).recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC1234" and out.alternative["text"] == "ABC1284"


class SuspectSvmOcr(StubSvmOcr):
    """An SVM that read 7 characters but flags its own segmentation as unreliable."""

    def recognize(self, image):
        out = super().recognize(image)
        return OcrOutput(raw_text=out.raw_text, confidence=None, detail=out.detail, engine=self.name, views={}, suspect=True)


def test_suspect_segmentation_hands_over_to_the_second_engine(recognizer):
    out = FallbackRecognizer(SuspectSvmOcr(), recognizer).recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC 1234" and out.engine == "fake-ocr"
    assert out.fallback is True and out.fallback_reason == "segmentation"
    assert out.alternative == {"engine": "svm-hog-chars", "text": "ABC1234", "confidence": None}


def test_suspect_segmentation_keeps_the_svm_if_the_other_engine_found_nothing(recognizer):
    recognizer.text = ""
    out = FallbackRecognizer(SuspectSvmOcr(), recognizer).recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC1234" and out.fallback is False


def test_unreadable_fallback_has_its_own_reason(recognizer):
    out = FallbackRecognizer(ShortSvmOcr(), recognizer).recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.fallback is True and out.fallback_reason == "unreadable"


def test_segmentation_suspect_flags_unsplit_pairs():
    from app.services.svm_ocr import Glyph, segmentation_suspect

    def g(w, h=40):
        return Glyph((0, 0, w, h), np.zeros((48, 24), np.uint8), w / h)

    assert not segmentation_suspect([g(20)] * 7)                              # regular plate
    assert not segmentation_suspect([g(14), g(37), g(16), g(30), g(30), g(36), g(40)])  # IF12YCC
    assert segmentation_suspect([g(26), g(28), g(40), g(20), g(25), g(41), g(21), g(39)])  # blurred touching pairs
    assert segmentation_suspect([g(20), g(20), g(60), g(20)])                 # aspect 1.5: cannot be one character
    assert not segmentation_suspect([g(10), g(40), g(10)])                    # too few glyphs to judge by median width


def test_fallback_recognizer_uses_second_engine_and_says_so(recognizer):
    fb = FallbackRecognizer(ShortSvmOcr(), recognizer)
    out = fb.recognize(np.zeros((10, 10, 3), np.uint8))
    assert out.raw_text == "ABC 1234" and out.engine == "fake-ocr" and out.fallback is True
    assert recognizer.calls == 1


def test_auto_mode_prefers_svm_and_reports_candidates(make_client, recognizer):
    client = make_client(svm_ocr=StubSvmOcr())
    img, _ = make_scene()
    r = post(client, encode(img)).json()["result"]              # default mode is auto
    assert r["ocr_engine"] == "svm-hog-chars" and r["ocr_fallback"] is False
    assert r["ocr_alternative"] == {"engine": "fake-ocr", "text": "ABC1234", "confidence": 0.9}
    assert r["candidates"] and sum(c["selected"] for c in r["candidates"]) == 1
    best = next(c for c in r["candidates"] if c["selected"])
    assert best["status"] == "readable" and best["text"] == "ABC1234" and best["crop"].startswith("data:image/png")
    assert [c["index"] for c in r["candidates"]] == list(range(1, len(r["candidates"]) + 1))


def test_auto_mode_falls_back_and_flags_it(make_client, recognizer):
    client = make_client(svm_ocr=ShortSvmOcr())
    img, _ = make_scene()
    r = post(client, encode(img)).json()["result"]
    assert r["ocr_engine"] == "fake-ocr" and r["ocr_fallback"] is True and recognizer.calls >= 1


def test_strict_modes_do_not_mix_engines(make_client, recognizer):
    client = make_client(svm_ocr=StubSvmOcr())
    img, _ = make_scene()
    assert post(client, encode(img), ocr_engine="rapidocr").json()["result"]["ocr_engine"] == "fake-ocr"
    before = recognizer.calls
    only_svm = post(client, encode(img), ocr_engine="svm").json()["result"]
    assert only_svm["ocr_fallback"] is False and only_svm["ocr_alternative"] is None
    assert recognizer.calls == before  # strict SVM mode never calls the other engine


def test_unread_candidate_is_reported_with_its_status(make_client):
    class Blank:
        name, language, ready = "blank", "en", True
        def status(self): return {"available": True, "enabled": True, "state": "ready", "detail": "x"}
        def is_available(self): return True, "x"
        def recognize(self, image): return OcrOutput(raw_text="", confidence=None)
    client = make_client(svm_ocr=Blank())
    img, _ = make_scene()
    r = post(client, encode(img), ocr_engine="svm").json()["result"]
    assert r["detection_status"] == "located_unreadable"
    assert r["candidates"][0]["status"] in {"small", "unread"} and r["candidates"][0]["text"] is None


# -------------------------------------------------- shipped model (if present)
@pytest.mark.skipif(not DEFAULT_SVM_OCR_MODEL.exists(), reason="no trained OCR model in backend/models")
@pytest.mark.parametrize("text", ["KLM5821", "ABC1234", "WXY3090", "H7Q2N60"])
def test_shipped_model_reads_clean_drawn_plates(text):
    r = SvmCharRecognizer(DEFAULT_SVM_OCR_MODEL)
    assert r.ready, r.status()
    assert r.recognize(prepare_for_ocr(draw_plate(text))).raw_text == text
