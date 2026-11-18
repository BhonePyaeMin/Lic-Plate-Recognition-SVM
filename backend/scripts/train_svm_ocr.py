"""Train the character SVM (36 classes, 0-9 A-Z) on synthetic plates.

An RBF-kernel SVM approximated with random Fourier features (default) or a plain linear SVM,
both trained with hinge loss by SGD on glyphs augmented with stroke weight, tilt, shear and width.

Run from the backend folder (needs scikit-learn, see requirements-train.txt):

    .venv/Scripts/python scripts/train_svm_ocr.py                 # full run
    .venv/Scripts/python scripts/train_svm_ocr.py --train 300 --val 60 --test 60 --compare 0 --out ../scratch

Data
----
Every plate is drawn by `synthetic_scenes.make_plate_crop` (random text over
all 36 characters, four fonts, four colour schemes, tilt, perspective, blur,
noise, JPEG), then passed through the SAME steps the server uses: `deskew`,
`prepare_for_ocr`, `segment_characters`. When the number of glyphs equals the
number of characters in the text, the glyphs are labelled left to right and
become training samples. Plates that fail segmentation contribute nothing to
training but DO count against the model in the held-out numbers.

Train, validation and test plates use disjoint seeds. Test plates are also drawn only with
TrueType typefaces held out of training (every fourth installed face), so the test numbers
measure unseen-typeface accuracy. The validation set picks C; the test set is only used for the
numbers written to the JSON file. Optionally
RapidOCR is run on the first --compare test plates for a side-by-side figure.

THE DATA IS SYNTHETIC. The metrics say nothing about real photographs.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

from app.services.image_preprocessing import deskew, prepare_for_ocr  # noqa: E402
from app.services.plate_recognizer import normalize_plate_text  # noqa: E402
from app.services.svm_ocr import (  # noqa: E402
    ALPHABET, GLYPH_FEATURE_DIM, Glyph, OcrSvmModel, glyph_features, normalize_glyph, segment_characters,
)
from synthetic_scenes import make_plate_crop, system_fonts  # noqa: E402

TRAIN_SEED0, VAL_SEED0, TEST_SEED0 = 500_000, 600_000, 700_000


def _init_worker():
    cv2.setNumThreads(1)


def font_split(seed: int) -> str:
    """Test plates only use typefaces the model never saw during training."""
    return "test" if seed >= TEST_SEED0 else "train"


AUG_PER_GLYPH = 3
_K3 = np.ones((3, 3), np.uint8)


def augment_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible variant of one glyph: heavier or lighter strokes, tilt, shear, narrower or wider.

    Real photographs differ from drawn text in exactly these ways (blur thickens strokes, thin
    fonts lose them, the camera tilts and skews), so the classifier should see them in training.
    """
    m = mask
    r = rng.random()
    if r < 0.25:
        m = cv2.dilate(m, _K3)
    elif r < 0.5 and min(m.shape) >= 9:
        m = cv2.erode(m, _K3)
    h, w = m.shape
    pad = max(h, w) // 3 + 2
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad : pad + h, pad : pad + w] = m
    ang = np.radians(rng.uniform(-7, 7))
    shear = rng.uniform(-0.12, 0.12)
    sx = rng.uniform(0.85, 1.15)
    A = np.array([[sx * np.cos(ang), np.sin(ang) + shear], [-np.sin(ang), np.cos(ang)]])
    c = np.array([canvas.shape[1] / 2, canvas.shape[0] / 2])
    M = np.hstack([A, (c - A @ c)[:, None]]).astype(np.float32)
    warped = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(warped > 100)
    if len(xs) < 6:
        return mask
    return ((warped[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 100) * 255).astype(np.uint8)


def process_plate(seed: int) -> dict:
    crop, text = make_plate_crop(seed, font_split(seed))
    target = "".join(c for c in text if c.isalnum())
    glyphs = segment_characters(prepare_for_ocr(deskew(crop)))
    feats = np.asarray([glyph_features(g) for g in glyphs], np.float32) if glyphs else np.zeros((0, GLYPH_FEATURE_DIM), np.float32)
    aug = np.zeros((0, GLYPH_FEATURE_DIM), np.float32)
    if seed < VAL_SEED0 and glyphs and len(glyphs) == len(target):  # training plates only
        rng = np.random.default_rng(seed + 11)
        variants = []
        for g in glyphs:
            for _ in range(AUG_PER_GLYPH):
                m = augment_mask(g.mask, rng)
                variants.append(glyph_features(Glyph(g.box, normalize_glyph(m), m.shape[1] / m.shape[0], m)))
        aug = np.asarray(variants, np.float32)  # glyph-major: AUG_PER_GLYPH rows per glyph, in order
    return {"seed": seed, "target": target, "n_glyphs": len(glyphs), "feats": feats, "aug": aug}


def build(seeds, pool) -> list[dict]:
    t = time.perf_counter()
    out = pool.map(process_plate, list(seeds), chunksize=16)
    print(f"  {len(out)} plates in {time.perf_counter() - t:.0f}s", flush=True)
    return out


def to_xy(plates: list[dict], augmented: bool = False):
    X, y = [], []
    for p in plates:
        if p["n_glyphs"] == len(p["target"]):
            labels = [ALPHABET.index(c) for c in p["target"]]
            X.append(p["feats"])
            y.extend(labels)
            if augmented and len(p["aug"]):
                X.append(p["aug"])
                y.extend(l for l in labels for _ in range(AUG_PER_GLYPH))
    return np.concatenate(X), np.asarray(y, np.int16)


def chain_metrics(plates: list[dict], predict) -> dict:
    """Plate-level results through the whole chain (segmentation + classifier)."""
    n = len(plates)
    seg_ok = exact = chars = chars_ok = 0
    confusions: Counter = Counter()
    for p in plates:
        if p["n_glyphs"] != len(p["target"]):
            continue
        seg_ok += 1
        pred = "".join(ALPHABET[i] for i in predict(p["feats"]))
        exact += pred == p["target"]
        for a, b in zip(p["target"], pred):
            chars += 1
            chars_ok += a == b
            if a != b:
                confusions[f"{a}->{b}"] += 1
    return {
        "plates": n,
        "segmentation_ok_rate": round(seg_ok / n, 4),
        "character_accuracy_when_segmented": round(chars_ok / chars, 4) if chars else None,
        "exact_match_rate_all_plates": round(exact / n, 4),
        "most_common_confusions": dict(confusions.most_common(8)),
    }


def rapidocr_comparison(count: int) -> dict | None:
    if count <= 0:
        return None
    try:
        from app.services.plate_recognizer import RapidOcrRecognizer
    except Exception:
        return None
    rec = RapidOcrRecognizer()
    ok, _ = rec.is_available()
    if not ok:
        return None
    exact = 0
    for seed in range(TEST_SEED0, TEST_SEED0 + count):
        crop, text = make_plate_crop(seed, font_split(seed))
        target = "".join(c for c in text if c.isalnum())
        exact += normalize_plate_text(rec.recognize(prepare_for_ocr(deskew(crop))).raw_text) == target
    return {"plates": count, "exact_match_rate": round(exact / count, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--compare", type=int, default=150, help="test plates also read by RapidOCR (0 = skip)")
    ap.add_argument("--kernel", choices=("rff", "linear"), default="rff", help="rff = RBF kernel via random Fourier features")
    ap.add_argument("--components", type=int, default=1500, help="random Fourier features (rff only)")
    ap.add_argument("--out", type=Path, default=BACKEND / "models")
    ap.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count() - 1)))
    args = ap.parse_args()

    from sklearn import __version__ as skl_version
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    print(f"Generating plates with {args.workers} workers", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        train = build(range(TRAIN_SEED0, TRAIN_SEED0 + args.train), pool)
        val = build(range(VAL_SEED0, VAL_SEED0 + args.val), pool)
        test = build(range(TEST_SEED0, TEST_SEED0 + args.test), pool)

    Xtr, ytr = to_xy(train, augmented=True)
    Xva, yva = to_xy(val)
    Xte, yte = to_xy(test)
    print(f"glyph samples  train {len(ytr)} (incl. {AUG_PER_GLYPH} augmented per glyph)  val {len(yva)}  test {len(yte)}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    mean = scaler.mean_.astype(np.float32)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)

    rff = None
    if args.kernel == "rff":
        rff = RBFSampler(gamma=1.0 / Ztr.shape[1], n_components=args.components, random_state=0).fit(Ztr[:10])
        Ptr, Pva = rff.transform(Ztr).astype(np.float32), rff.transform(Zva).astype(np.float32)
    else:
        Ptr, Pva = Ztr, Zva

    best = None
    for alpha in (1e-6, 1e-5):
        clf = SGDClassifier(loss="hinge", alpha=alpha, max_iter=30, tol=None, n_jobs=2, random_state=0).fit(Ptr, ytr)
        acc = float((clf.predict(Pva) == yva).mean())
        print(f"  alpha={alpha:<7} val glyph accuracy {acc:.4f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, alpha, clf)
    _, alpha, clf = best
    print(f"chosen alpha={alpha}")

    model = OcrSvmModel(
        ALPHABET, mean, scale, clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32), {},
        None if rff is None else rff.random_weights_.astype(np.float32), None if rff is None else rff.random_offset_.astype(np.float32),
    )

    def predict(feats: np.ndarray) -> np.ndarray:  # the exact arithmetic the server runs
        return np.argmax(model.scores(feats), axis=1)

    meta = {
        "model": ("glyph HOG (24x48) + 6x12 thumbnail + aspect; RBF-kernel SVM approximated with "
                  f"{args.components} random Fourier features + 36-class linear SVM (hinge loss, SGD)") if rff is not None
                 else "glyph HOG (24x48) + 6x12 thumbnail + aspect, 36-class linear SVM (hinge loss, SGD)",
        "trained_on": "synthetic drawn plates only (no real photographs)",
        "created": dt.date.today().isoformat(),
        "scikit_learn": skl_version,
        "feature_dim": GLYPH_FEATURE_DIM,
        "classes": ALPHABET,
        "alpha": alpha,
        "augmentation": f"{AUG_PER_GLYPH} variants per training glyph: stroke weight, rotation, shear, width",
        "plates": {"train": args.train, "val": args.val, "test": args.test},
        "fonts": {
            "train": ["opencv-hershey x4"] + [Path(f).stem for f in system_fonts("train")],
            "held_out_test": [Path(f).stem for f in system_fonts("test")] or ["(none found: test plates use the Hershey fonts)"],
        },
        "glyph_samples": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "seeds": {"train_from": TRAIN_SEED0, "val_from": VAL_SEED0, "test_from": TEST_SEED0},
        "validation_glyph_accuracy": round(float((predict(Xva) == yva).mean()), 4),
        "test_glyph_accuracy": round(float((predict(Xte) == yte).mean()), 4),
        "test_through_whole_chain": chain_metrics(test, predict),
        "caveat": "All numbers are on synthetic plates. Real-photograph accuracy has not been measured.",
    }
    comp = rapidocr_comparison(args.compare)
    if comp:
        meta["rapidocr_same_synthetic_plates"] = comp

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = dict(classes=np.array([ALPHABET]), mean=mean, scale=scale, coef=model.coef, intercept=model.intercept)
    if rff is not None:
        arrays.update(rff_w=model.rff_w, rff_b=model.rff_b)
    np.savez_compressed(args.out / "ocr_svm.npz", **arrays)
    (args.out / "ocr_svm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k.startswith(("validation", "test", "rapidocr"))}, indent=2))
    print(f"wrote {args.out / 'ocr_svm.npz'}")


if __name__ == "__main__":
    main()
