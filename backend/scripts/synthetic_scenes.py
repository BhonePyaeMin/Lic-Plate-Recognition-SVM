"""Randomised synthetic car scenes with known plate boxes, for SVM training.

Everything here is DRAWN with OpenCV primitives. A model trained on these
scenes has never seen a photograph; see the limitation in app/services/svm_plate.py.

`make_random_scene(seed, with_plate)` returns (image_bgr, plate_box_or_None)
where the box is (x, y, w, h). The same seed always gives the same scene.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
CHARS = list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
SIZES = [(720, 1080), (600, 900), (540, 960), (800, 1200), (480, 720)]
# (background, text, border) in BGR: white, yellow, black, light blue-grey plates.
PLATE_STYLES = [
    ((245, 245, 245), (20, 20, 20), (30, 30, 30)),
    ((70, 200, 235), (20, 20, 20), (30, 30, 30)),
    ((25, 25, 25), (240, 240, 240), (200, 200, 200)),
    ((235, 225, 210), (60, 30, 20), (90, 60, 40)),
]


ALL_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # includes I, O, Q for character training


# TrueType faces that look like plate lettering (sans, mono, DIN-style). Only the files found on the
# training machine are used, and only their rendered shapes influence the model: no font is
# distributed. Every fourth face (sorted) is held out of training to measure unseen-typeface accuracy.
_TTF_NAMES = [
    "arial", "arialbd", "arialn", "arialnb", "bahnschrift", "calibri", "calibrib", "calibril", "candara", "candarab", "candaral",
    "consola", "consolab", "corbel", "corbelb", "corbell", "cour", "courbd", "dubai-bold", "dubai-light", "dubai-medium",
    "dubai-regular", "ebrima", "ebrimabd", "frabk", "fradm", "fradmcn", "framd", "framdcn", "frahv", "gadugi", "gadugib",
    "gil_____", "gilb____", "gilsanub", "gothic", "gothicb", "agencyb", "agencyr", "leelawui", "lsans", "lsansd", "lucon",
    "malgun", "malgunbd", "malgunsl", "micross", "ocraext", "refsan", "segoeui", "segoeuib", "segoeuil", "segoeuisl", "seguisb",
    "tahoma", "tahomabd", "tcb_____", "tcm_____", "tccm____", "trebuc", "trebucbd", "verdana", "verdanab", "brlnsr", "erasmd",
    "eraslght", "maian",
]
_FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu"),
              Path("/usr/share/fonts/truetype/liberation"), Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]


def system_fonts(split: str = "train") -> list[str]:
    """TrueType files for `split` = 'train' (3 of 4 faces), 'test' (the held-out quarter) or 'all'."""
    found = sorted({str((d / f"{n}.ttf").resolve()).lower() for d in _FONT_DIRS for n in _TTF_NAMES if (d / f"{n}.ttf").exists()})
    if split == "all":
        return found
    held_out = [f for i, f in enumerate(found) if i % 4 == 3]
    return held_out if split == "test" else [f for f in found if f not in held_out]


def random_plate_text(rng: np.random.Generator, chars=CHARS) -> str:
    n = int(rng.integers(5, 9))
    text = "".join(rng.choice(chars, n))
    if rng.random() < 0.35:
        cut = int(rng.integers(2, n - 1))
        text = text[:cut] + str(rng.choice([" ", "-"])) + text[cut:]
    return text


def _rand_color(rng: np.random.Generator, lo: int = 20, hi: int = 235) -> tuple[int, int, int]:
    return tuple(int(v) for v in rng.integers(lo, hi, 3))


def _background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    top, bottom = np.array(_rand_color(rng, 90, 235)), np.array(_rand_color(rng, 40, 200))
    t = np.linspace(0, 1, h)[:, None, None]
    img[:] = (top * (1 - t) + bottom * t).astype(np.uint8)
    return img


def _clutter(img: np.ndarray, rng: np.random.Generator) -> None:
    """Non-plate structure. Includes plate-shaped, edge-rich patches (grilles, fences, stripes)."""
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(4, 10))):
        kind = rng.choice(["body", "window", "grille", "stripes", "lights", "fence", "blocks", "bar"])
        x, y = int(rng.integers(0, w * 3 // 4)), int(rng.integers(0, h * 3 // 4))
        bw, bh = int(rng.integers(w // 10, w // 2)), int(rng.integers(h // 14, h // 3))
        x1, y1 = min(w - 1, x + bw), min(h - 1, y + bh)
        if kind == "body":
            cv2.rectangle(img, (x, y), (x1, y1), _rand_color(rng), -1)
        elif kind == "window":
            cv2.rectangle(img, (x, y), (x1, y1), _rand_color(rng, 20, 110), -1)
            cv2.line(img, (x, y), (x1, y1), _rand_color(rng, 150, 235), int(rng.integers(1, 4)))
        elif kind == "grille":  # dense horizontal bars: edge-rich and plate-shaped
            cv2.rectangle(img, (x, y), (x1, y1), _rand_color(rng, 20, 90), -1)
            step = int(rng.integers(4, 10))
            for yy in range(y, y1, step):
                cv2.line(img, (x, yy), (x1, yy), _rand_color(rng, 120, 230), int(rng.integers(1, 3)))
        elif kind == "stripes":
            step = int(rng.integers(5, 16))
            c1, c2 = _rand_color(rng), _rand_color(rng)
            for xx in range(x, x1, step):
                cv2.rectangle(img, (xx, y), (min(x1, xx + step // 2), y1), c1 if (xx // step) % 2 else c2, -1)
        elif kind == "lights":
            for cx in (x, x1):
                cv2.circle(img, (cx, y + bh // 2), int(rng.integers(12, 36)), _rand_color(rng, 180, 255), -1)
        elif kind == "fence":  # crossing lines
            c = _rand_color(rng, 30, 200)
            step = int(rng.integers(8, 20))
            for xx in range(x, x1, step):
                cv2.line(img, (xx, y), (xx + step, y1), c, 1)
                cv2.line(img, (xx + step, y), (xx, y1), c, 1)
        elif kind == "blocks":  # tiles / brick
            step_y = int(rng.integers(8, 20))
            for yy in range(y, y1, step_y):
                off = int(rng.integers(0, step_y * 2))
                for xx in range(x - off, x1, step_y * 2):
                    cv2.rectangle(img, (max(x, xx), yy), (min(x1, xx + step_y * 2 - 2), min(y1, yy + step_y - 2)), _rand_color(rng, 60, 220), -1)
        else:  # bar: bumper / trim line
            cv2.rectangle(img, (x, y), (x1, y + max(3, bh // 5)), _rand_color(rng, 10, 120), -1)


def _draw_plate(rng: np.random.Generator, text: str, pw: int, ph: int, ttf_split: str | None = None) -> np.ndarray:
    """Plate patch (BGR, ph x pw) with border and fitted text.

    `ttf_split` None keeps the original OpenCV Hershey fonts only (used by the scene generator so
    the plate-detector training data is unchanged). 'train' / 'test' / 'all' also allow TrueType
    faces (see `system_fonts`), chosen about 70 percent of the time when any exist.
    """
    bg, fg, border = PLATE_STYLES[int(rng.integers(len(PLATE_STYLES)))]
    patch = np.full((ph, pw, 3), bg, np.uint8)
    cv2.rectangle(patch, (0, 0), (pw - 1, ph - 1), border, max(1, ph // 18))
    # Real plates often carry a blue country strip on the left and a line of tiny print below the
    # characters (dealer or regulation text). Both are drawn only for the character-training plates.
    left = 0
    if ttf_split and rng.random() < 0.3:
        left = int(pw * rng.uniform(0.09, 0.14))
        cv2.rectangle(patch, (0, 0), (left, ph), (150, 70, 20), -1)
        cv2.putText(patch, "RO", (max(1, left // 6), int(ph * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, max(0.25, ph / 130), (255, 255, 255), 1, cv2.LINE_AA)
    if ttf_split and rng.random() < 0.3:
        small = "".join(rng.choice(list("ABCDEFGHIJKLMNOPRSTUVW0123456789 "), 28))
        cv2.putText(patch, small, (left + int(pw * 0.12), int(ph * 0.95)), cv2.FONT_HERSHEY_PLAIN, max(0.3, ph / 160), (150, 150, 150), 1, cv2.LINE_AA)
    fonts = system_fonts(ttf_split) if ttf_split else []
    if fonts and (ttf_split == "test" or rng.random() < 0.8):
        path = fonts[int(rng.integers(len(fonts)))]
        size = max(8, int(ph * 0.7))
        img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for _ in range(2):  # measure, rescale to fit, measure again
            font = ImageFont.truetype(path, size)
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            size = max(8, int(size * min(0.84 * (pw - left) / (r - l), 0.56 * ph / (b - t))))
        font = ImageFont.truetype(path, size)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        draw.text((left + (pw - left - (r - l)) / 2 - l, (ph - (b - t)) / 2 - t - ph * 0.03), text, font=font, fill=(fg[2], fg[1], fg[0]))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    font = FONTS[int(rng.integers(len(FONTS)))]
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    scale = min(0.84 * pw / tw, 0.58 * ph / th)
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(patch, text, ((pw - tw) // 2, (ph + th) // 2), font, scale, fg, thick, cv2.LINE_AA)
    return patch


def make_random_scene(seed: int, with_plate: bool = True):
    rng = np.random.default_rng(seed)
    h, w = SIZES[int(rng.integers(len(SIZES)))]
    img = _background(rng, h, w)
    _clutter(img, rng)
    box = None

    if with_plate:
        pw = int(rng.uniform(0.09, 0.30) * w)
        ph = max(20, int(pw / rng.uniform(2.6, 5.2)))
        patch = _draw_plate(rng, random_plate_text(rng), pw, ph)
        # Place with room for rotation; keep it fully inside the frame.
        margin = int(0.15 * pw)
        x = int(rng.integers(margin, max(margin + 1, w - pw - margin)))
        y = int(rng.integers(margin, max(margin + 1, h - ph - margin)))
        layer = np.zeros_like(img)
        mask = np.zeros((h, w), np.uint8)
        layer[y : y + ph, x : x + pw] = patch
        mask[y : y + ph, x : x + pw] = 255
        # Rotation plus a mild perspective jitter of the four corners.
        ang = float(rng.uniform(-10, 10))
        rot = cv2.getRotationMatrix2D((x + pw / 2, y + ph / 2), ang, 1.0)
        src = np.float32([[x, y], [x + pw, y], [x + pw, y + ph], [x, y + ph]])
        dst = cv2.transform(src[None], rot)[0] + rng.uniform(-0.04, 0.04, (4, 2)).astype(np.float32) * [pw, ph]
        warp = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        layer = cv2.warpPerspective(layer, warp, (w, h))
        mask = cv2.warpPerspective(mask, warp, (w, h))
        img[mask > 0] = layer[mask > 0]
        xs, ys, bw, bh = cv2.boundingRect(mask)
        box = (int(xs), int(ys), int(bw), int(bh))

    # Camera effects.
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    alpha, beta = float(rng.uniform(0.7, 1.3)), float(rng.uniform(-35, 35))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
    img = np.clip(img + rng.normal(0, rng.uniform(0, 14), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), box


def make_plate_crop(seed: int, ttf_split: str | None = None):
    """One drawn plate as a detector crop would look, plus its text.

    Returns (image_bgr, text). The text may contain a space or hyphen; strip
    those for character labels. The crop includes a margin of background and
    the same kinds of distortion as `make_random_scene`.
    """
    rng = np.random.default_rng(seed)
    text = random_plate_text(rng, ALL_CHARS)
    pw = int(rng.uniform(100 if ttf_split else 140, 360))
    ph = max(32, int(pw / rng.uniform(2.6, 5.2)))
    patch = _draw_plate(rng, text, pw, ph, ttf_split)
    mx, my = int(pw * rng.uniform(0.04, 0.16)), int(ph * rng.uniform(0.1, 0.35))
    h, w = ph + 2 * my, pw + 2 * mx
    canvas = np.empty((h, w, 3), np.uint8)
    canvas[:] = _rand_color(rng, 40, 210)
    canvas[my : my + ph, mx : mx + pw] = patch
    src = np.float32([[mx, my], [mx + pw, my], [mx + pw, my + ph], [mx, my + ph]])
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-8, 8)), 1.0)
    dst = cv2.transform(src[None], rot)[0] + rng.uniform(-0.03, 0.03, (4, 2)).astype(np.float32) * [pw, ph]
    warp = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    img = cv2.warpPerspective(canvas, warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if ttf_split and rng.random() < 0.6:  # resolution loss: shrink then enlarge, as a small distant plate would look
        k = float(rng.uniform(0.3, 0.8))
        small = cv2.resize(img, (max(24, int(w * k)), max(10, int(h * k))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])))
    if ttf_split and rng.random() < 0.35:  # motion blur
        length = int(rng.integers(3, 12))
        kern = np.zeros((length, length), np.float32)
        kern[length // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), float(rng.uniform(-10, 10)), 1.0)
        kern = cv2.warpAffine(kern, rot, (length, length))
        img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = img.astype(np.float32)
    if ttf_split:
        img = 255.0 * (img / 255.0) ** float(rng.uniform(0.6, 1.7))  # gamma: night shots, glare
        shade = np.linspace(float(rng.uniform(0.7, 1.0)), float(rng.uniform(0.7, 1.0)), w)[None, :, None]
        img = img * shade  # uneven lighting across the plate
    img = np.clip(img * float(rng.uniform(0.7, 1.3)) + float(rng.uniform(-35, 35)), 0, 255)
    img = np.clip(img + rng.normal(0, rng.uniform(0, 12), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), text
