"""SVM character recognition: an alternative OCR engine.

Pipeline (all classical, no neural network)
-------------------------------------------
1. `segment_characters` binarises the plate image (both ink polarities are
   tried, the one giving the larger plausible glyph count wins), labels connected components, keeps those that look like characters
   (height close to the median, not touching the border, not the plate frame),
   splits obviously merged pairs at the weakest vertical-projection column and
   sorts the glyphs left to right.
2. Each glyph is scaled to fit a 24x48 canvas with its aspect ratio kept, and
   described by HOG features (shared `hog` from svm_plate), a 6x12 thumbnail
   and the glyph aspect ratio: 433 numbers.
3. A multi-class SVM (one-vs-rest, 36 classes 0-9 A-Z) picks the class with the
   highest score. The shipped model is an RBF-kernel SVM approximated with random
   Fourier features (a linear SVM on 1500 cosine features); a plain linear model
   also loads. The score is a margin, not a probability.

Model file `models/ocr_svm.npz` holds `classes`, `mean`, `scale`, `coef`,
`intercept` as plain arrays (no pickle); it is produced by
`backend/scripts/train_svm_ocr.py`. Serving needs only numpy and OpenCV.

Limitations (read before trusting it)
-------------------------------------
* Trained only on synthetic, drawn plates in four OpenCV Hershey fonts. Real
  plate typefaces, dirt, blur and glare have not been tested.
* Characters that touch, break apart or are joined to the frame can be
  mis-segmented; when no character is found the engine returns no text.
* 0/O, 1/I and 8/B are close in shape; no context rules repair them.
* Single-row plates only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.services.plate_recognizer import MIN_TEXT_CHARS, OcrOutput, normalize_plate_text
from app.services.svm_plate import hog

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLYPH_W, GLYPH_H = 24, 48
_FIT_W, _FIT_H = 20, 42            # glyph is fitted inside this box, centred on the canvas
MIN_GLYPHS, MAX_GLYPHS = 3, 12     # a plate reading outside this range is treated as a failed segmentation
GLYPH_FEATURE_DIM = 360 + 72 + 1   # HOG(24x48) + 6x12 thumbnail + aspect ratio
SPLIT_VALLEY: float | None = 0.3   # split a wide glyph where ink drops below this share of its peak; None = never split
SPLIT_FORCE_ASPECT = 1.35          # wider than this (width / height) cannot be one character: always split
SPLIT_WIDE_RATIO = 1.85            # blobs this many times the typical width may split on a shallower valley ...
SPLIT_VALLEY_WIDE = 0.6            # ... this share of the peak (blur fills the gap between touching characters)
SUSPECT_ASPECT = 1.35              # a glyph still wider than this after splitting means segmentation is unreliable
SUSPECT_WIDTH_RATIO = 1.45         # ... and so does one this many times wider than the median glyph (blurred, touching pairs)


@dataclass(frozen=True)
class Glyph:
    box: tuple[int, int, int, int]  # x, y, w, h in the segmented image
    image: np.ndarray               # GLYPH_H x GLYPH_W uint8, ink = 255
    aspect: float                   # width / height of the glyph box
    mask: Optional[np.ndarray] = None  # raw ink mask cropped to the box (used by training augmentation)


# ------------------------------------------------------------ segmentation
def _components(binary: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Character-like connected components as (box, ink mask cropped to the box)."""
    h_img, w_img = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        # Loose absolute limits: crops carry wide margins, so characters can be only
        # ~15 percent of the image height. The median-height check below does the real work.
        if area < 0.002 * h_img * w_img or h < 0.10 * h_img:
            continue
        if h > 0.9 * h_img or w > 0.6 * w_img:          # plate frame or car body, not a character
            continue
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:  # touching the border
            continue
        if w / h > 3.2:
            continue
        cands.append(((x, y, w, h), (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255))
    if not cands:
        return []
    med_h = float(np.median([c[0][3] for c in cands]))
    med_cy = float(np.median([c[0][1] + c[0][3] / 2 for c in cands]))
    kept = [c for c in cands
            if 0.6 * med_h <= c[0][3] <= 1.4 * med_h and abs(c[0][1] + c[0][3] / 2 - med_cy) < 0.5 * med_h]
    if not kept:
        return []

    # Split merged characters: a box much wider than a typical one is cut at the
    # weakest column of its ink projection near each expected boundary.
    narrow = [c[0][2] for c in kept if c[0][2] / c[0][3] < 0.95] or [c[0][2] for c in kept]
    typical_w = float(np.median(narrow))
    out = []
    for (x, y, w, h), mask in kept:
        # A blob much wider than a typical character may be several touching characters. A lone
        # wide letter (W, M, X) must survive, so the decision looks at the ink projection:
        #  * aspect beyond SPLIT_FORCE_ASPECT cannot be one character: split at the weakest columns;
        #  * otherwise split only at a clear valley (SPLIT_VALLEY), or a shallower one
        #    (SPLIT_VALLEY_WIDE) when the blob is far wider than usual, as blur fills the gap.
        parts = None
        if w > 1.7 * typical_w and SPLIT_VALLEY is not None:
            k = max(2, int(round(w / (1.1 * typical_w))))
            proj = mask.sum(axis=0).astype(np.float32)
            cuts = [0]
            for j in range(1, k):
                centre = int(round(j * w / k))
                lo, hi = max(cuts[-1] + 2, centre - max(2, w // (4 * k))), min(w - 2, centre + max(2, w // (4 * k)))
                cuts.append(int(lo + np.argmin(proj[lo : hi + 1])) if hi > lo else centre)
            cuts.append(w)
            depth = max(proj[c] for c in cuts[1:-1]) / max(1.0, float(proj.max()))
            if w / h > SPLIT_FORCE_ASPECT or depth <= SPLIT_VALLEY or (w > SPLIT_WIDE_RATIO * typical_w and depth <= SPLIT_VALLEY_WIDE):
                parts = list(zip(cuts[:-1], cuts[1:]))
        if parts:
            for a, b in parts:
                part = mask[:, a:b]
                ys, xs = np.nonzero(part)
                if len(xs) == 0:
                    continue
                px0, px1, py0, py1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                out.append(((x + a + int(px0), y + int(py0), int(px1 - px0), int(py1 - py0)), part[py0:py1, px0:px1]))
        else:
            out.append(((x, y, w, h), mask))
    return sorted(out, key=lambda c: c[0][0])


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Fit an ink mask into a GLYPH_H x GLYPH_W canvas, keeping its aspect ratio."""
    h, w = mask.shape
    scale = min(_FIT_W / w, _FIT_H / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLYPH_H, GLYPH_W), np.uint8)
    y0, x0 = (GLYPH_H - nh) // 2, (GLYPH_W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def segment_characters(image: np.ndarray) -> list[Glyph]:
    """Split a plate image into ordered character glyphs (empty list if nothing plausible)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Local threshold: copes with uneven lighting and with strokes that a global (Otsu)
    # threshold breaks apart on small, blurred plates.
    block = max(15, (gray.shape[0] // 3) | 1)
    adaptive_dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 8)
    def reading(inks) -> list:
        """Best plausible reading over both polarities of the given binary images: the most glyphs."""
        found: list = []
        for ink in inks:
            light = 255 - ink
            for binary in ((ink, light) if ink.mean() <= 127 else (light, ink)):
                comps = _components(binary)
                if MIN_GLYPHS <= len(comps) <= MAX_GLYPHS and len(comps) > len(found):
                    found = comps
        return found

    def wide(comps) -> int:  # glyphs that are still too wide to be one character
        return sum(1 for c in comps if c[0][2] / c[0][3] > SUSPECT_ASPECT)

    # The global (Otsu) reading is trusted first. The local threshold is only consulted when
    # that reading still contains over-wide glyphs; it then wins only if it has fewer of them,
    # so its extra noisy blobs cannot replace a clean Otsu reading.
    best = reading([dark_ink])
    if best and wide(best):  # never on its own: from pure noise the local threshold invents "characters"
        alt = reading([adaptive_dark])
        if alt and wide(alt) < wide(best):
            best = alt
    return [Glyph(box, normalize_glyph(mask), box[2] / box[3], mask) for box, mask in best]


# ------------------------------------------------------------------- views
def render_views(image: np.ndarray, glyphs: list[Glyph], chars: list[str]) -> dict:
    """Images that show what the SVM did: the segmented plate and the glyph strip.

    `segmented`: the image given to the recognizer with a green box and the
    predicted character around every glyph. `glyphs`: the normalised 24x48
    glyphs exactly as the classifier received them (ink black on white,
    enlarged 2x) with the predicted character underneath.
    """
    seg = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thickness = max(1, seg.shape[0] // 120)
    for g, ch in zip(glyphs, chars):
        x, y, w, h = g.box
        cv2.rectangle(seg, (x, y), (x + w, y + h), (0, 190, 0), thickness)
        cv2.putText(seg, ch, (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, seg.shape[0] / 300), (0, 140, 0), thickness, cv2.LINE_AA)
    cw, ch_ = GLYPH_W * 2, GLYPH_H * 2
    strip = np.full((ch_ + 26, max(1, len(glyphs)) * (cw + 8) + 8, 3), 255, np.uint8)
    for i, (g, c) in enumerate(zip(glyphs, chars)):
        tile = 255 - cv2.resize(g.image, (cw, ch_), interpolation=cv2.INTER_NEAREST)
        x0 = 8 + i * (cw + 8)
        strip[4 : 4 + ch_, x0 : x0 + cw] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(strip, c, (x0 + cw // 2 - 7, ch_ + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return {"segmented": seg, "glyphs": strip}


# ---------------------------------------------------------------- features
def segmentation_suspect(glyphs: list[Glyph]) -> bool:
    """True when a glyph still looks too wide to be one character (touching characters left unsplit).

    Two signals: an aspect ratio no single character has, or a width far above the plate's median
    glyph width (a blurred pair such as "20" can have an ordinary aspect ratio but is twice as wide
    as its neighbours). Unsplit pairs make the character count and the reading wrong.
    """
    if any(g.aspect > SUSPECT_ASPECT for g in glyphs):
        return True
    if len(glyphs) >= 4:
        widths = [g.box[2] for g in glyphs]
        return max(widths) > SUSPECT_WIDTH_RATIO * float(np.median(widths))
    return False


def glyph_features(glyph: Glyph) -> np.ndarray:
    img = cv2.GaussianBlur(glyph.image, (3, 3), 0)
    thumb = cv2.resize(glyph.image, (6, 12), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255.0
    return np.concatenate([hog(img), thumb, np.float32([min(glyph.aspect, 3.0)])]).astype(np.float32)


# ------------------------------------------------------------------- model
@dataclass(frozen=True)
class OcrSvmModel:
    """Multi-class SVM. Linear on the glyph features, or (when `rff_w` is set) an RBF-kernel SVM
    approximated with random Fourier features: phi(z) = sqrt(2/D) * cos(z W + b), then a linear SVM."""

    classes: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray        # (n_classes, GLYPH_FEATURE_DIM) linear, or (n_classes, D) with random features
    intercept: np.ndarray   # (n_classes,)
    meta: dict
    rff_w: Optional[np.ndarray] = None  # (GLYPH_FEATURE_DIM, D)
    rff_b: Optional[np.ndarray] = None  # (D,)

    def scores(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.scale
        if self.rff_w is not None:
            z = np.sqrt(2.0 / self.rff_w.shape[1]) * np.cos(z @ self.rff_w + self.rff_b)
        return z @ self.coef.T + self.intercept


def load_ocr_model(path: Path) -> OcrSvmModel:
    with np.load(path, allow_pickle=False) as d:
        classes = str(d["classes"].reshape(-1)[0])
        mean, scale, coef, intercept = d["mean"], d["scale"], d["coef"], d["intercept"]
        rff_w = d["rff_w"] if "rff_w" in d.files else None
        rff_b = d["rff_b"] if "rff_b" in d.files else None
    width = GLYPH_FEATURE_DIM if rff_w is None else rff_w.shape[1]
    if (mean.shape != (GLYPH_FEATURE_DIM,) or coef.shape != (len(classes), width)
            or (rff_w is not None and (rff_w.shape[0] != GLYPH_FEATURE_DIM or rff_b is None or rff_b.shape != (width,)))):
        raise ValueError(f"model shape {coef.shape} does not match the {GLYPH_FEATURE_DIM}-value glyph features")
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return OcrSvmModel(
        classes, mean.astype(np.float32), np.where(scale == 0, 1.0, scale).astype(np.float32),
        coef.astype(np.float32), intercept.astype(np.float32), meta,
        None if rff_w is None else rff_w.astype(np.float32), None if rff_b is None else rff_b.astype(np.float32),
    )


class SvmCharRecognizer:
    """`PlateRecognizer` backed by segmentation plus a multi-class SVM."""

    name = "svm-hog-chars"
    language = "en"

    def __init__(self, model_path: Optional[Path], enabled: bool = True):
        self.enabled = enabled
        self._model: Optional[OcrSvmModel] = None
        self._state, self._detail = "disabled", "SVM character recognition is turned off (ENABLE_SVM_OCR=false)."
        if not enabled:
            return
        if model_path is None or not Path(model_path).exists():
            self._state = "no_model"
            self._detail = "No trained model found. Run backend/scripts/train_svm_ocr.py to create models/ocr_svm.npz."
            return
        try:
            self._model = load_ocr_model(Path(model_path))
        except Exception as exc:
            self._state, self._detail = "failed", f"Could not load the SVM character model ({type(exc).__name__})."
            return
        self._state = "ready"
        self._detail = f"Character SVM (HOG), trained on {self._model.meta.get('trained_on', 'unknown data')}."

    @classmethod
    def disabled(cls) -> "SvmCharRecognizer":
        return cls(None, enabled=False)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {"available": self.ready, "enabled": self.enabled, "state": self._state, "detail": self._detail}

    def is_available(self) -> tuple[bool, str]:
        return self.ready, self._detail

    def classify(self, glyph: Glyph) -> tuple[str, float]:
        """Best class and its margin (top score) for one glyph."""
        if self._model is None:
            raise RuntimeError("SVM character model is not loaded")
        s = self._model.scores(glyph_features(glyph))
        i = int(np.argmax(s))
        return self._model.classes[i], float(s[i])

    def recognize(self, image: np.ndarray) -> OcrOutput:
        glyphs = segment_characters(image)
        if not glyphs:
            return OcrOutput(raw_text="", confidence=None)
        detail = []
        for g in glyphs:
            ch, margin = self.classify(g)
            detail.append({"char": ch, "margin": round(margin, 3), "x": g.box[0], "y": g.box[1], "width": g.box[2], "height": g.box[3]})
        # The margin is not a probability, so no confidence is reported.
        text = "".join(d["char"] for d in detail)
        return OcrOutput(raw_text=text, confidence=None, detail=detail, engine=self.name,
                         views=render_views(image, glyphs, [d["char"] for d in detail]),
                         suspect=segmentation_suspect(glyphs))


class FallbackRecognizer:
    """SVM first; the second engine reads only when the SVM cannot, and otherwise cross-checks.

    "Cannot read" means fewer than MIN_TEXT_CHARS letters or digits after
    normalisation (no characters found, or segmentation produced too few). The
    SVM's views are kept so the UI can still show what it attempted, and the
    output is flagged `fallback=True` so nothing pretends the SVM read it.
    """

    name = "svm-hog-chars+rapidocr-fallback"
    language = "en"

    def __init__(self, primary: SvmCharRecognizer, secondary):
        self.primary, self.secondary = primary, secondary

    def is_available(self) -> tuple[bool, str]:
        ok_p, _ = self.primary.is_available()
        ok_s, detail = self.secondary.is_available()
        return ok_p or ok_s, detail

    def recognize(self, image: np.ndarray) -> OcrOutput:
        out = self.primary.recognize(image)
        svm_text = normalize_plate_text(out.raw_text)
        if len(svm_text) >= MIN_TEXT_CHARS:
            # The SVM produced a plausible reading. Ask the second engine too, as a cross-check.
            try:
                other = self.secondary.recognize(image)
                other_text = normalize_plate_text(other.raw_text)
                alternative = {"engine": getattr(self.secondary, "name", None), "text": other_text or None, "confidence": other.confidence}
            except Exception:  # a failing second opinion must never break a good SVM reading
                return out
            if out.suspect and len(other_text) >= MIN_TEXT_CHARS:
                # The SVM's own character splitting looked unreliable (touching characters it could
                # not separate), so its reading is probably wrong. Use the second engine's text, and
                # keep the SVM's reading visible as the alternative.
                return OcrOutput(raw_text=other.raw_text, confidence=other.confidence, detail=None,
                                 engine=getattr(self.secondary, "name", None), views=out.views, fallback=True,
                                 fallback_reason="segmentation",
                                 alternative={"engine": self.primary.name, "text": svm_text, "confidence": None})
            return replace(out, alternative=alternative)
        alt = self.secondary.recognize(image)
        return OcrOutput(raw_text=alt.raw_text, confidence=alt.confidence, detail=None,
                         engine=getattr(self.secondary, "name", None), views=out.views, fallback=True,
                         fallback_reason="unreadable")
