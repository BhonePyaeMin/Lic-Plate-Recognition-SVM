"""Upload validation, decoding and image enhancement helpers.

All functions are pure (numpy in, numpy out) except `validate_and_decode`,
which only touches the bytes it is given. Nothing is written to disk and the
client filename is used solely to check its extension.
"""
from __future__ import annotations

import io
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.core.errors import (
    EmptyUploadError,
    FileTooLargeError,
    InvalidImageError,
    UnsupportedFileError,
)

# Pillow format name -> acceptable lowercase extensions / MIME types.
_FORMATS = {
    "JPEG": ({"jpg", "jpeg"}, {"image/jpeg", "image/jpg", "image/pjpeg"}),
    "PNG": ({"png"}, {"image/png"}),
    "WEBP": ({"webp"}, {"image/webp"}),
}


def validate_and_decode(
    data: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    settings: Settings,
) -> np.ndarray:
    """Validate an upload and return a BGR uint8 image.

    Signals used, in order: size, emptiness, extension, MIME type (advisory,
    only rejected when it is clearly not an image), file signature (Pillow
    header sniff), pixel count, and finally a full OpenCV decode.
    """
    if not data:
        raise EmptyUploadError("The uploaded file is empty. Choose an image and try again.")
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"The image is larger than the {settings.max_upload_mb} MB limit. "
            "Resize or compress it and try again."
        )

    ext = os.path.splitext(os.path.basename(filename or ""))[1].lower().lstrip(".")
    allowed = ", ".join(e.upper() for e in settings.allowed_extensions)
    if ext not in settings.allowed_extensions:
        raise UnsupportedFileError(f"Unsupported file type. Upload one of: {allowed}.")

    # MIME is client-controlled, so it can only veto, never approve.
    if content_type and not content_type.lower().startswith(("image/", "application/octet-stream")):
        raise UnsupportedFileError(f"Unsupported file type. Upload one of: {allowed}.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size  # header only, no pixel decode
    except (UnidentifiedImageError, OSError, ValueError):
        raise InvalidImageError(
            "The file could not be read as an image. It may be corrupted; try another file."
        ) from None

    if fmt not in _FORMATS or ext not in _FORMATS[fmt][0]:
        raise UnsupportedFileError(
            f"The file contents do not match its extension or are not a supported format ({allowed})."
        )
    if width * height > settings.max_image_pixels:
        raise InvalidImageError(
            f"The image has too many pixels ({width}x{height}). Downscale it and try again."
        )

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError(
            "The image data is corrupted and could not be decoded. Try another file."
        )
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def denoise(gray: np.ndarray) -> np.ndarray:
    # Bilateral filtering smooths texture (asphalt, paint grain) while keeping
    # the strong character edges that the detector depends on.
    return cv2.bilateralFilter(gray, d=9, sigmaColor=17, sigmaSpace=17)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE equalises locally, so a plate in shadow is not washed out by a
    # bright sky the way global histogram equalisation would do.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_side. Returns (image, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def prepare_for_ocr(crop: np.ndarray, target_height: int = 96) -> np.ndarray:
    """Upscale, contrast-enhance and pad a plate crop for the OCR engine.

    Returns a 3-channel image (the OCR engine expects colour or gray input).
    Small crops are upscaled with cubic interpolation because character
    strokes only a few pixels wide are otherwise lost; large crops are left
    alone to avoid pointless work.
    """
    gray = to_gray(crop)
    h, w = gray.shape[:2]
    if h < target_height:
        scale = target_height / h
        gray = cv2.resize(gray, (int(round(w * scale)), target_height), interpolation=cv2.INTER_CUBIC)
    gray = enhance_contrast(gray)
    pad = max(8, gray.shape[0] // 8)  # OCR detectors miss text touching the image border
    border = int(np.median(gray))
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def deskew(crop: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate a plate crop so its text baseline is horizontal.

    The skew is estimated from the minimum-area rectangle around the dark
    (or light) character pixels after Otsu thresholding. Rotations smaller
    than 0.5 degrees or larger than `max_angle` are ignored: the first is
    noise, the second is more likely a bad estimate than a real tilt.
    """
    gray = to_gray(crop)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Characters should be the minority class; flip if the mask is mostly set.
    if binary.mean() > 127:
        binary = 255 - binary
    # Drop components touching the crop border (plate frame, screws, edge of
    # the car body) and specks; what remains should be the characters.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ch, cw = binary.shape
    min_area = 0.002 * ch * cw
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w_, h_, area = stats[i]
        touches = x <= 0 or y <= 0 or x + w_ >= cw or y + h_ >= ch
        keep[i] = (not touches) and area >= min_area
    ys, xs = np.nonzero(keep[labels])
    if len(xs) < 50:
        return crop
    (_, _), (rw, rh), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    # minAreaRect angle convention differs across OpenCV versions; fold the
    # result into (-45, 45] relative to the long side.
    if rw < rh:
        angle -= 90.0
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return crop
    h, w = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
