"""Train the HOG + linear SVM plate verifier on synthetic scenes.

Run from the backend folder (needs scikit-learn: pip install -r requirements-train.txt):

    .venv/Scripts/python scripts/train_svm.py                # full run, a few minutes
    .venv/Scripts/python scripts/train_svm.py --train 60 --val 20 --test 20 --out ../scratch   # smoke test

Data
----
Each synthetic scene (scripts/synthetic_scenes.py) is run through the real
`ContourPlateDetector`. Every candidate it proposes becomes a training sample:
IoU >= 0.5 with the true plate box = positive, IoU < 0.3 = negative (the
ambiguous 0.3 to 0.5 band is skipped). The negatives are therefore the
detector's own false alarms (grilles, fences, stripes), which is exactly the
mistake the SVM has to learn to reject. Two jittered copies of each true box
(shifted and padded) are added as extra positives, and three random plate-
shaped boxes per scene as extra negatives.

Train, validation and test scenes use disjoint random seeds. The validation
set picks C and the suggested rejection threshold; the test set is only used
for the final numbers written to the model's JSON file.

THE DATA IS SYNTHETIC. The metrics say nothing about real photographs.
"""
from __future__ import annotations

import os

# One BLAS thread per process: each worker otherwise reserves large per-thread
# buffers, which fails on small-memory machines. Must be set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

from app.services.plate_detector import ContourPlateDetector  # noqa: E402
from app.services.svm_plate import FEATURE_DIM, SvmPlateDetector, extract_features, squash  # noqa: E402
from synthetic_scenes import make_random_scene  # noqa: E402

TRAIN_SEED0, VAL_SEED0, TEST_SEED0 = 0, 100_000, 200_000
POS_IOU, NEG_IOU = 0.5, 0.3


def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def _init_worker():
    cv2.setNumThreads(1)


def process_scene(seed: int) -> dict:
    """Generate one scene, run the baseline detector, return features and metadata."""
    with_plate = (seed % 10) >= 3  # 30 percent of scenes have no plate at all
    img, gt = make_random_scene(seed, with_plate)
    rng = np.random.default_rng(seed + 7)
    h, w = img.shape[:2]
    cands = []
    for c in ContourPlateDetector().detect(img):
        crop = img[c.y : c.y + c.height, c.x : c.x + c.width]
        if crop.size == 0:
            continue
        cands.append({"box": (c.x, c.y, c.width, c.height), "score": c.score, "feat": extract_features(crop),
                      "iou": iou((c.x, c.y, c.width, c.height), gt) if gt else 0.0})
    extra = []  # (feature, label)
    if gt:
        gx, gy, gw, gh = gt
        for _ in range(2):  # jittered true boxes: what a slightly-off proposal looks like
            padx, pady = rng.uniform(0, 0.15) * gw, rng.uniform(0, 0.3) * gh
            sx, sy = rng.uniform(-0.06, 0.06) * gw, rng.uniform(-0.1, 0.1) * gh
            x0, y0 = int(max(0, gx - padx + sx)), int(max(0, gy - pady + sy))
            x1, y1 = int(min(w, gx + gw + padx + sx)), int(min(h, gy + gh + pady + sy))
            if iou((x0, y0, x1 - x0, y1 - y0), gt) >= POS_IOU:
                extra.append((extract_features(img[y0:y1, x0:x1]), 1))
    for _ in range(3):  # random plate-shaped boxes away from the plate
        bw = int(rng.uniform(0.08, 0.3) * w)
        bh = max(16, int(bw / rng.uniform(2.0, 6.0)))
        x0, y0 = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        if gt is None or iou((x0, y0, bw, bh), gt) < 0.1:
            extra.append((extract_features(img[y0 : y0 + bh, x0 : x0 + bw]), 0))
    return {"seed": seed, "gt": gt, "cands": cands, "extra": extra}


def build(seeds, pool) -> list[dict]:
    t = time.perf_counter()
    out = pool.map(process_scene, list(seeds), chunksize=8)
    print(f"  {len(out)} scenes in {time.perf_counter() - t:.0f}s", flush=True)
    return out


def to_xy(scenes: list[dict]):
    X, y = [], []
    for s in scenes:
        for c in s["cands"]:
            if c["iou"] >= POS_IOU:
                X.append(c["feat"]); y.append(1)
            elif c["iou"] < NEG_IOU:
                X.append(c["feat"]); y.append(0)
        for f, label in s["extra"]:
            X.append(f); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int8)


def end_to_end(scenes, margin_fn, min_margin: float) -> dict:
    """Top-1 hit rate on plate scenes and false-alarm rate on no-plate scenes."""
    base_hit = svm_hit = base_any = svm_any = plates = nop = base_fa = svm_fa = 0
    for s in scenes:
        cands = s["cands"]
        scored = []
        for c in cands:
            m = margin_fn(c["feat"])
            if m >= min_margin:
                scored.append((SvmPlateDetector.HEURISTIC_WEIGHT * c["score"] + (1 - SvmPlateDetector.HEURISTIC_WEIGHT) * squash(m), c))
        scored.sort(key=lambda t: t[0], reverse=True)
        base_sorted = sorted(cands, key=lambda c: c["score"], reverse=True)
        if s["gt"]:
            plates += 1
            base_any += any(c["iou"] >= POS_IOU for c in cands)
            svm_any += any(c["iou"] >= POS_IOU for _, c in scored)
            base_hit += bool(base_sorted) and base_sorted[0]["iou"] >= POS_IOU
            svm_hit += bool(scored) and scored[0][1]["iou"] >= POS_IOU
        else:
            nop += 1
            base_fa += bool(cands)
            svm_fa += bool(scored)
    r = lambda a, n: round(a / n, 4) if n else None  # noqa: E731
    return {
        "plate_scenes": plates, "no_plate_scenes": nop,
        "baseline_top1_hit_rate": r(base_hit, plates), "svm_top1_hit_rate": r(svm_hit, plates),
        "baseline_any_candidate_hit_rate": r(base_any, plates), "svm_any_candidate_hit_rate": r(svm_any, plates),
        "baseline_false_alarm_rate_no_plate": r(base_fa, nop), "svm_false_alarm_rate_no_plate": r(svm_fa, nop),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=1200, help="training scenes")
    ap.add_argument("--val", type=int, default=300, help="validation scenes")
    ap.add_argument("--test", type=int, default=300, help="held-out test scenes")
    ap.add_argument("--out", type=Path, default=BACKEND / "models", help="output folder")
    ap.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count() - 1)),
                    help="parallel scene workers (lower this if memory is tight)")
    args = ap.parse_args()

    from sklearn import __version__ as skl_version
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    print(f"Generating scenes with {args.workers} workers", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        train = build(range(TRAIN_SEED0, TRAIN_SEED0 + args.train), pool)
        val = build(range(VAL_SEED0, VAL_SEED0 + args.val), pool)
        test = build(range(TEST_SEED0, TEST_SEED0 + args.test), pool)

    Xtr, ytr = to_xy(train)
    Xva, yva = to_xy(val)
    Xte, yte = to_xy(test)
    print(f"samples  train {len(ytr)} (pos {int(ytr.sum())})  val {len(yva)} (pos {int(yva.sum())})  test {len(yte)} (pos {int(yte.sum())})")
    assert Xtr.shape[1] == FEATURE_DIM

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LinearSVC(C=C, class_weight="balanced", dual=False, max_iter=10_000).fit(Ztr, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Zva))
        print(f"  C={C:<6} val AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    _, C, clf = best
    print(f"chosen C={C}")

    def metrics(Z, y):
        d = clf.decision_function(Z)
        p, r, f, _ = precision_recall_fscore_support(y, d > 0, average="binary", zero_division=0)
        return {"samples": int(len(y)), "positives": int(y.sum()), "accuracy": round(float(accuracy_score(y, d > 0)), 4),
                "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
                "roc_auc": round(float(roc_auc_score(y, d)), 4)}

    # Suggested threshold: keep 97 percent of validation true plates.
    val_pos_margins = clf.decision_function(Zva)[yva == 1]
    suggested = float(min(0.0, np.percentile(val_pos_margins, 3)))

    w, b = clf.coef_.reshape(-1), float(clf.intercept_[0])
    mean, scale = scaler.mean_.astype(np.float32), np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    margin_fn = lambda f: float(np.dot((f - mean) / scale, w) + b)  # noqa: E731

    meta = {
        "model": "HOG (128x32, 8px cells, 9 bins, 2x2 blocks, L2-Hys) + LinearSVC",
        "trained_on": "synthetic drawn scenes only (no real photographs)",
        "created": dt.date.today().isoformat(),
        "scikit_learn": skl_version,
        "feature_dim": FEATURE_DIM,
        "C": C,
        "class_weight": "balanced",
        "scenes": {"train": args.train, "val": args.val, "test": args.test},
        "seeds": {"train_from": TRAIN_SEED0, "val_from": VAL_SEED0, "test_from": TEST_SEED0},
        "suggested_min_margin": round(suggested, 4),
        "validation": metrics(Zva, yva),
        "test_candidates": metrics(Zte, yte),
        "test_end_to_end": end_to_end(test, margin_fn, suggested),
        "caveat": "All numbers are on synthetic scenes. Real-photograph accuracy has not been measured.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "plate_svm.npz", mean=mean, scale=scale, coef=w.astype(np.float32), intercept=np.float32(b))
    (args.out / "plate_svm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("suggested_min_margin", "validation", "test_candidates", "test_end_to_end")}, indent=2))
    print(f"wrote {args.out / 'plate_svm.npz'}")


if __name__ == "__main__":
    main()
