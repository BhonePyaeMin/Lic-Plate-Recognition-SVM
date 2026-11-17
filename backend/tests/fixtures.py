"""Synthetic test scenes. No real photographs are bundled with the repo.

`make_scene` draws a simple road/car-body background with a plate whose
position is known, so detector output can be compared with ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_scene(
    text: str = "ABC1234",
    size: tuple[int, int] = (720, 1080),
    plate_xywh: tuple[int, int, int, int] = (400, 430, 280, 70),
    tilt_deg: float = 0.0,
    noise: float = 6.0,
    seed: int = 0,
    with_plate: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), 150, np.uint8)
    img[: h // 2] = (205, 190, 170)  # sky
    img[h * 3 // 4 :] = (95, 95, 95)  # road
    # car body and window
    cv2.rectangle(img, (w // 5, h // 4), (w * 4 // 5, h * 3 // 4), (60, 50, 140), -1)
    cv2.rectangle(img, (w // 4, h // 3), (w * 3 // 4, h // 2), (90, 80, 70), -1)
    # headlights and grille to add non-plate structure
    for cx in (w // 3, w * 2 // 3):
        cv2.circle(img, (cx, h // 2 + 60), 28, (220, 235, 240), -1)
    if with_plate:
        x, y, pw, ph = plate_xywh
        layer = np.zeros_like(img)
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(layer, (x, y), (x + pw, y + ph), (245, 245, 245), -1)
        cv2.rectangle(layer, (x, y), (x + pw, y + ph), (30, 30, 30), 2)
        scale = ph / 70 * 1.55
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        cv2.putText(layer, text, (x + (pw - tw) // 2, y + (ph + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.rectangle(mask, (x, y), (x + pw, y + ph), 255, -1)
        if tilt_deg:
            m = cv2.getRotationMatrix2D((x + pw / 2, y + ph / 2), tilt_deg, 1.0)
            layer = cv2.warpAffine(layer, m, (w, h))
            mask = cv2.warpAffine(mask, m, (w, h))
        img[mask > 0] = layer[mask > 0]
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img, plate_xywh


def encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)
