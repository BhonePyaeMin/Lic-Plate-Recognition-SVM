"""Generate synthetic demo images (no real photographs are bundled).

Run from the repo root:  backend/.venv/Scripts/python samples/generate_samples.py
Writes JPEGs next to this script. They are drawn scenes, so results on them say
nothing about accuracy on real photographs.
"""
import sys
from pathlib import Path

import cv2

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "backend"))
from tests.fixtures import make_scene  # noqa: E402

cases = {
    "synthetic_front.jpg": dict(text="ABC1234"),
    "synthetic_tilted.jpg": dict(text="KLM5821", tilt_deg=7),
    "synthetic_noisy.jpg": dict(text="WXY3090", noise=18, tilt_deg=-4),
    "synthetic_no_plate.jpg": dict(with_plate=False),
}
for name, kw in cases.items():
    img, _ = make_scene(**kw)
    cv2.imwrite(str(here / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", name)
