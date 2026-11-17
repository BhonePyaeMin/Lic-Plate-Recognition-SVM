"""User-defined preprocessing: crop, then a chain of catalog operations."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.errors import InvalidParamsError, MatlabUnavailableError
from app.services.filters import CATALOG, resolve_params, to_bgr
from app.services.matlab_adapter import MatlabAdapter

MAX_STEPS = 8
MIN_CROP_W, MIN_CROP_H = 24, 12
ENGINES = ("auto", "opencv", "matlab")


@dataclass
class Step:
    id: str
    params: dict


def _loads(raw: Optional[str], what: str):
    if raw is None or raw.strip() == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise InvalidParamsError(f"The {what} setting is not valid JSON.") from None


def parse_engine(raw: Optional[str]) -> str:
    engine = (raw or "auto").lower()
    if engine not in ENGINES:
        raise InvalidParamsError("Engine must be auto, opencv or matlab.")
    return engine


def parse_steps(raw: Optional[str]) -> list[Step]:
    data = _loads(raw, "filter steps")
    if not data:
        return []
    if not isinstance(data, list) or len(data) > MAX_STEPS:
        raise InvalidParamsError(f"Use at most {MAX_STEPS} filter steps.")
    steps = []
    for item in data:
        if not isinstance(item, dict) or item.get("id") not in CATALOG:
            raise InvalidParamsError("Unknown filter in the step list.")
        steps.append(Step(item["id"], resolve_params(CATALOG[item["id"]], item.get("params"))))
    return steps


def parse_crop(raw: Optional[str], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    data = _loads(raw, "crop")
    if not data:
        return None
    try:
        x, y, w, h = (int(round(float(data[k]))) for k in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        raise InvalidParamsError("The crop must contain x, y, width and height.") from None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < MIN_CROP_W or y1 - y0 < MIN_CROP_H:
        raise InvalidParamsError(f"The crop area is too small (minimum {MIN_CROP_W} x {MIN_CROP_H} px) or outside the image.")
    return x0, y0, x1 - x0, y1 - y0


def apply_crop(image: np.ndarray, crop: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return image
    x, y, w, h = crop
    return np.ascontiguousarray(image[y : y + h, x : x + w])


def run_steps(
    image: np.ndarray,
    steps: list[Step],
    engine: str,
    matlab: MatlabAdapter,
    ref: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run steps in order. Returns the BGR result and one record per step."""
    records: list[dict] = []
    current = image
    for step in steps:
        op = CATALOG[step.id]
        if op.needs_reference and ref is None:
            raise InvalidParamsError(f"{op.label} needs a reference image. Choose one before running.")
        use_matlab = False
        if engine == "matlab":
            if not op.matlab:
                raise MatlabUnavailableError(f"{op.label} is only implemented with OpenCV.")
            use_matlab = True
        elif engine == "auto":
            use_matlab = op.matlab and matlab.ready

        started = time.perf_counter()
        used, fallback = "opencv", False
        result = None
        if use_matlab:
            try:
                result = matlab.run_operation(step.id, step.params, current, ref)
                used = "matlab"
            except MatlabUnavailableError:
                if engine == "matlab":
                    raise
                fallback = True  # auto mode: fall back rather than fail the request
        if result is None:
            result = op.run(current, step.params, ref)
        current = to_bgr(result)
        records.append({
            "id": step.id, "label": op.label, "engine": used, "fallback": fallback,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "output": op.output,
        })
    return current, records
