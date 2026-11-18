"""Catalog of image operations for the three course modules (OpenCV/NumPy side).

Every operation has a stable `id`. The MATLAB dispatcher
(`matlab/api/run_operation_file.m`) uses the same ids and parameter keys, so
the same request can be executed by either engine. Results are equivalent in
intent, not bit-identical (pixel-centre conventions and library details
differ).

Operations take a BGR uint8 image and return a uint8 image (gray or BGR).
Segmentation operations return a 0/255 mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.core.errors import InvalidParamsError

INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
MAX_OUTPUT_PIXELS = 25_000_000


# --------------------------------------------------------------------------
# Catalog types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # "number" | "select"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: tuple = ()

    def public(self) -> dict:
        d = {"key": self.key, "label": self.label, "kind": self.kind, "default": self.default}
        if self.kind == "number":
            d.update(min=self.min, max=self.max, step=self.step)
        else:
            d["options"] = list(self.options)
        return d


@dataclass(frozen=True)
class Operation:
    id: str
    module: int
    group: str
    label: str
    description: str
    params: tuple
    run: Callable[[np.ndarray, dict, Optional[np.ndarray]], np.ndarray]
    output: str = "image"  # "image" or "mask" (binary result, not suitable as OCR input)
    needs_reference: bool = False
    matlab: bool = True  # False: only the OpenCV engine implements it

    def public(self) -> dict:
        return {
            "id": self.id, "module": self.module, "group": self.group, "label": self.label,
            "description": self.description, "output": self.output,
            "needs_reference": self.needs_reference, "matlab": self.matlab,
            "params": [p.public() for p in self.params],
        }


def num(key, label, default, lo, hi, step=1):
    return Param(key, label, "number", default, lo, hi, step)


def sel(key, label, default, options):
    return Param(key, label, "select", default, options=tuple(options))


INTERP_PARAM = sel("interpolation", "Interpolation", "bilinear", ["nearest", "bilinear", "bicubic"])
MAPPING_PARAM = sel("mapping", "Mapping", "reverse", ["reverse", "forward"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _mask(m: np.ndarray) -> np.ndarray:
    return (m.astype(np.uint8) * 255) if m.dtype == bool else np.where(m > 0, 255, 0).astype(np.uint8)


def _clip8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)


def _odd(k: float) -> int:
    k = int(round(k))
    return k if k % 2 else k + 1


def _per_channel(img: np.ndarray, fn) -> np.ndarray:
    if img.ndim == 2:
        return fn(img)
    return np.dstack([fn(img[:, :, c]) for c in range(img.shape[2])])


def _check_size(h: int, w: int) -> None:
    if h < 1 or w < 1 or h * w > MAX_OUTPUT_PIXELS:
        raise InvalidParamsError("The requested output size is too large. Use a smaller scale factor or crop first.")


# --------------------------------------------------------------------------
# Module 1: enhancement
# --------------------------------------------------------------------------
def _hist_eq(img, p, ref):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # V only: keeps colours intact
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    cs = np.cumsum(np.bincount(src.ravel(), minlength=256)) / src.size
    cr = np.cumsum(np.bincount(ref.ravel(), minlength=256)) / ref.size
    lut = np.clip(np.searchsorted(cr, cs - 1e-12), 0, 255).astype(np.uint8)
    return lut[src]


def _hist_match(img, p, ref):
    if ref is None:
        raise InvalidParamsError("Histogram matching needs a reference image. Choose one in the step settings.")
    return np.dstack([_match_channel(img[:, :, c], ref[:, :, c]) for c in range(3)])


def _brightness(img, p, ref):
    return _clip8(p["contrast"] * (img.astype(np.float32) - 128.0) + 128.0 + p["brightness"])


def _gamma(img, p, ref):
    return _clip8(255.0 * (img.astype(np.float32) / 255.0) ** p["gamma"])


def _mean(img, p, ref):
    k = int(round(p["kernel"]))
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REPLICATE)


def _median(img, p, ref):
    return cv2.medianBlur(img, _odd(p["kernel"]))


def _laplacian(img, p, ref):
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REPLICATE)  # [0 1 0;1 -4 1;0 1 0]
    return _clip8(f - p["strength"] * lap)


def _highpass(img, p, ref):
    f = img.astype(np.float32)
    k = int(round(p["kernel"]))
    low = cv2.blur(f, (k, k), borderType=cv2.BORDER_REPLICATE)
    return _clip8(f + p["boost"] * (f - low))


def _fourier_channel(ch: np.ndarray, kind: str, d1: float, d2: float, shape: str) -> np.ndarray:
    f = ch.astype(np.float64)
    M, N = f.shape
    P, Q = 2 * M, 2 * N  # zero padding avoids wrap-around artefacts
    F = np.fft.fftshift(np.fft.fft2(f, s=(P, Q)))
    v, u = np.mgrid[0:P, 0:Q]
    D = np.sqrt((u - Q / 2) ** 2 + (v - P / 2) ** 2)

    def lowpass(d0):
        return (D <= d0).astype(np.float64) if shape == "ideal" else np.exp(-(D ** 2) / (2 * d0 ** 2))

    H = {"low": lambda: lowpass(d1), "high": lambda: 1 - lowpass(d1), "band": lambda: lowpass(d2) - lowpass(d1)}[kind]()
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))[:M, :N]
    if kind == "low":
        return _clip8(g)
    span = g.max() - g.min()  # high/band remove DC: rescale for display
    return _clip8((g - g.min()) / span * 255.0) if span > 0 else np.zeros_like(ch)


def _fourier(kind):
    def run(img, p, ref):
        d1 = p.get("cutoff", p.get("inner"))
        d2 = p.get("outer", 0)
        return _per_channel(img, lambda ch: _fourier_channel(ch, kind, d1, d2, p["shape"]))
    return run


# --------------------------------------------------------------------------
# Module 2: segmentation, edges, morphology
# --------------------------------------------------------------------------
def _gradient_edges(gray: np.ndarray, kx: np.ndarray, thr: float) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    gx = cv2.filter2D(g, -1, kx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(g, -1, kx.T, borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag /= mag.max()
    return _mask(mag >= thr)


_SOBEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
_PREWITT = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)


def _sobel(img, p, ref):
    return _gradient_edges(_gray(img), _SOBEL, p["threshold"])


def _prewitt(img, p, ref):
    return _gradient_edges(_gray(img), _PREWITT, p["threshold"])


def _canny(img, p, ref):
    return cv2.Canny(cv2.GaussianBlur(_gray(img), (5, 5), 1.4), int(p["low"] * 255), int(p["high"] * 255))


def _otsu(img, p, ref):
    _, bw = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw


def _region_growing(img, p, ref):
    gray = _gray(img)
    h, w = gray.shape
    seed = (int(np.clip(p["seed_x"] / 100.0 * (w - 1), 0, w - 1)), int(np.clip(p["seed_y"] / 100.0 * (h - 1), 0, h - 1)))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tol = int(round(p["tolerance"]))
    # Pixels join when within `tol` gray levels of the seed value, 8-connected.
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(gray.copy(), mask, seed, 255, tol, tol, flags)
    return mask[1:-1, 1:-1]


def _rgb_seg(img, p, ref):
    lo = np.array([p["b_min"], p["g_min"], p["r_min"]], np.uint8)  # OpenCV is BGR
    hi = np.array([p["b_max"], p["g_max"], p["r_max"]], np.uint8)
    return cv2.inRange(img, lo, hi)


def _hsv_seg(img, p, ref):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)  # H scaled to 0..255 -> degrees * 255/360
    h = hsv[:, :, 0].astype(np.float32) * 360.0 / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    h0, h1 = p["h_min"], p["h_max"]
    hue = (h >= h0) & (h <= h1) if h0 <= h1 else (h >= h0) | (h <= h1)  # wrap-around for red
    return _mask(hue & (s >= p["s_min"]) & (s <= p["s_max"]) & (v >= p["v_min"]) & (v <= p["v_max"]))


def _binary_input(img: np.ndarray) -> np.ndarray:
    gray = _gray(img)
    if set(np.unique(gray)) <= {0, 255}:
        return gray
    return _otsu(gray, {}, None)


def _morph(op):
    def run(img, p, ref):
        size = max(1, int(round(p["size"])))
        if p["shape"] == "disk":
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
        else:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, max(1, round(size / 2))))
        return cv2.morphologyEx(_binary_input(img), op, se)
    return run


def _plate_shapes(img, p, ref):
    """Keep blobs that look like a plate: vertical-edge density, closed horizontally, aspect and extent filtered."""
    gray = _gray(img)
    h, w = gray.shape
    gx = np.abs(cv2.filter2D(gray.astype(np.float32) / 255.0, -1, _SOBEL, borderType=cv2.BORDER_REPLICATE))
    edges = _otsu(_clip8(255 * gx / max(gx.max(), 1e-9)), {}, None)
    kw = max(9, round(w / 40))
    fused = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    out = np.zeros_like(gray)
    min_area = 0.0005 * h * w
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area or ch == 0:
            continue
        aspect = cw / ch
        if p["min_aspect"] <= aspect <= p["max_aspect"] and area / (cw * ch) >= 0.4:
            out[labels == i] = 255
    return out


# --------------------------------------------------------------------------
# Module 3: geometry
# --------------------------------------------------------------------------
def _warp_forward(img: np.ndarray, M: np.ndarray, size: tuple) -> np.ndarray:
    """Forward mapping: push each source pixel to round(M * p). Leaves holes and collisions by design."""
    h, w = img.shape[:2]
    out_h, out_w = size
    ys, xs = np.mgrid[0:h, 0:w]
    u = np.rint(M[0, 0] * xs + M[0, 1] * ys + M[0, 2]).astype(int)
    v = np.rint(M[1, 0] * xs + M[1, 1] * ys + M[1, 2]).astype(int)
    ok = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    out = np.zeros((out_h, out_w) + img.shape[2:], img.dtype)
    out[v[ok], u[ok]] = img[ys[ok], xs[ok]]
    return out


def _affine(img, M, size, interpolation, mapping):
    _check_size(*size)
    if mapping == "forward":
        return _warp_forward(img, M, size)
    return cv2.warpAffine(img, M, (size[1], size[0]), flags=INTERP[interpolation], borderMode=cv2.BORDER_CONSTANT)


def _translate(img, p, ref):
    M = np.array([[1, 0, p["tx"]], [0, 1, p["ty"]]], np.float64)
    return _affine(img, M, img.shape[:2], p["interpolation"], p["mapping"])


def _scale(img, p, ref):
    h, w = img.shape[:2]
    if p.get("mode", "enlarge") == "zoom-in":
        # Magnify (or shrink) about a fixed point and keep the frame size, so the plate can be
        # brought up close without the picture growing. The point (cx, cy) stays where it is.
        f = p["factor"]
        cx, cy = p.get("center_x", 50) / 100.0 * (w - 1), p.get("center_y", 50) / 100.0 * (h - 1)
        M = np.float32([[f, 0, cx * (1 - f)], [0, f, cy * (1 - f)]])
        return cv2.warpAffine(img, M, (w, h), flags=INTERP[p["interpolation"]], borderMode=cv2.BORDER_CONSTANT)
    # Round half UP, as MATLAB does. Python's round() rounds halves to even, so 493 * 0.5 = 246.5 became
    # 246 here but 247 in MATLAB, and the same filter produced different image sizes in the two engines.
    nh, nw = max(1, int(np.floor(h * p["factor"] + 0.5))), max(1, int(np.floor(w * p["factor"] + 0.5)))
    _check_size(nh, nw)
    return cv2.resize(img, (nw, nh), interpolation=INTERP[p["interpolation"]])


def _rotate(img, p, ref):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), p["angle"], 1.0)  # positive angle = counter-clockwise
    size = (h, w)
    if p["enlarge"] == "yes":
        c, s = abs(M[0, 0]), abs(M[0, 1])
        size = (int(np.ceil(h * c + w * s)), int(np.ceil(h * s + w * c)))
        M[0, 2] += (size[1] - w) / 2.0
        M[1, 2] += (size[0] - h) / 2.0
    return _affine(img, M, size, p["interpolation"], p["mapping"])


def _shear(img, p, ref):
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = np.array([[1, p["shear_x"], -p["shear_x"] * cy], [p["shear_y"], 1, -p["shear_y"] * cx]], np.float64)
    return _affine(img, M, (h, w), p["interpolation"], "reverse")


def _radial(img, p, ref):
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    norm = np.hypot(cx, cy)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xu, yu = (u - cx) / norm, (v - cy) / norm
    r2 = xu ** 2 + yu ** 2
    factor = 1 + p["k1"] * r2 + p["k2"] * r2 ** 2  # distorted = undistorted * factor
    return cv2.remap(img, (xu * factor * norm + cx).astype(np.float32), (yu * factor * norm + cy).astype(np.float32),
                     INTERP[p["interpolation"]], borderMode=cv2.BORDER_CONSTANT)


def _register(img, p, ref):
    """Feature-based registration: ORB keypoints, brute-force matching, RANSAC homography."""
    if ref is None:
        raise InvalidParamsError("Registration needs a reference image to align to. Choose one in the step settings.")
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(_gray(img), None)
    k2, d2 = orb.detectAndCompute(_gray(ref), None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        raise InvalidParamsError("Not enough image features found to align the images.")
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:200]
    if len(matches) < 10:
        raise InvalidParamsError("Too few matching features to align the images. Try images of the same scene.")
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers is None or int(inliers.sum()) < 8:
        raise InvalidParamsError("The images could not be aligned reliably. Try images of the same scene.")
    rh, rw = ref.shape[:2]
    return cv2.warpPerspective(img, H, (rw, rh), flags=INTERP[p["interpolation"]])


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def _ops() -> list[Operation]:
    G1, G2, G3 = "Histogram", "Spatial filtering", "Frequency domain"
    shape = sel("shape", "Filter shape", "gaussian", ["gaussian", "ideal"])
    morph_params = (num("size", "Size (px)", 5, 1, 41), sel("shape", "Structuring element", "rectangle", ["rectangle", "disk"]))
    return [
        # Module 1
        Operation("hist_equalization", 1, G1, "Histogram equalization", "Spreads gray levels evenly; lifts contrast on dull or hazy photos.", (), _hist_eq),
        Operation("hist_matching", 1, G1, "Histogram matching", "Gives the image the tonal distribution of a reference image.", (), _hist_match, needs_reference=True),
        Operation("brightness_contrast", 1, G1, "Brightness and contrast", "Linear gray-level transform about mid-gray.",
                  (num("contrast", "Contrast", 1.3, 0.1, 3, 0.05), num("brightness", "Brightness", 0, -128, 128, 1)), _brightness),
        Operation("gamma", 1, G1, "Gamma correction", "Power-law transform. Below 1 brightens shadows.", (num("gamma", "Gamma", 0.7, 0.1, 4, 0.05),), _gamma),
        Operation("mean", 1, G2, "Mean smoothing", "Box blur for noise reduction.", (num("kernel", "Kernel size", 5, 1, 31),), _mean),
        Operation("median", 1, G2, "Median smoothing", "Removes salt-and-pepper noise and keeps edges.", (num("kernel", "Kernel size", 3, 1, 31),), _median),
        Operation("laplacian", 1, G2, "Laplacian sharpening", "Subtracts the Laplacian to restore edge detail.", (num("strength", "Strength", 1, 0, 3, 0.1),), _laplacian),
        Operation("highpass", 1, G2, "High-pass / high-boost sharpening", "Adds back the difference from a blurred copy.",
                  (num("kernel", "Blur size", 5, 3, 31), num("boost", "Boost", 1.5, 0, 5, 0.1)), _highpass),
        Operation("fourier_low", 1, G3, "Fourier low-pass", "Removes high frequencies (noise, fine detail).", (num("cutoff", "Cut-off radius", 40, 1, 400), shape), _fourier("low")),
        Operation("fourier_high", 1, G3, "Fourier high-pass", "Keeps edges and detail, removes smooth areas.", (num("cutoff", "Cut-off radius", 30, 1, 400), shape), _fourier("high")),
        Operation("fourier_band", 1, G3, "Fourier band-pass", "Keeps a ring of frequencies between two radii.",
                  (num("inner", "Inner radius", 20, 1, 400), num("outer", "Outer radius", 80, 2, 800), shape), _fourier("band")),
        # Module 2
        Operation("sobel", 2, "Edge detection", "Sobel edges", "Gradient magnitude with Sobel kernels.", (num("threshold", "Threshold", 0.25, 0.01, 1, 0.01),), _sobel, output="mask"),
        Operation("prewitt", 2, "Edge detection", "Prewitt edges", "Gradient magnitude with Prewitt kernels.", (num("threshold", "Threshold", 0.25, 0.01, 1, 0.01),), _prewitt, output="mask"),
        Operation("canny", 2, "Edge detection", "Canny edges", "Smoothing, non-maximum suppression and hysteresis.",
                  (num("low", "Low threshold", 0.1, 0.01, 1, 0.01), num("high", "High threshold", 0.3, 0.02, 1, 0.01)), _canny, output="mask"),
        Operation("otsu", 2, "Segmentation", "Otsu threshold", "Automatic global threshold.", (), _otsu, output="mask"),
        Operation("region_growing", 2, "Segmentation", "Region growing", "Grows a region from a seed over similar gray levels.",
                  (num("seed_x", "Seed x (%)", 50, 0, 100), num("seed_y", "Seed y (%)", 50, 0, 100), num("tolerance", "Tolerance", 20, 1, 128)), _region_growing, output="mask"),
        Operation("rgb_seg", 2, "Color segmentation", "RGB range segmentation", "Selects pixels inside an RGB box.",
                  (num("r_min", "Red min", 170, 0, 255), num("r_max", "Red max", 255, 0, 255), num("g_min", "Green min", 170, 0, 255),
                   num("g_max", "Green max", 255, 0, 255), num("b_min", "Blue min", 170, 0, 255), num("b_max", "Blue max", 255, 0, 255)), _rgb_seg, output="mask"),
        Operation("hsv_seg", 2, "Color segmentation", "HSV range segmentation", "Selects by hue (degrees), saturation and value.",
                  (num("h_min", "Hue min", 0, 0, 360), num("h_max", "Hue max", 360, 0, 360), num("s_min", "Saturation min", 0, 0, 1, 0.01),
                   num("s_max", "Saturation max", 0.2, 0, 1, 0.01), num("v_min", "Value min", 0.65, 0, 1, 0.01), num("v_max", "Value max", 1, 0, 1, 0.01)), _hsv_seg, output="mask"),
        Operation("dilate", 2, "Morphology", "Dilation", "Grows white regions. Non-binary input is Otsu-thresholded first.", morph_params, _morph(cv2.MORPH_DILATE), output="mask"),
        Operation("erode", 2, "Morphology", "Erosion", "Shrinks white regions.", morph_params, _morph(cv2.MORPH_ERODE), output="mask"),
        Operation("open", 2, "Morphology", "Opening", "Erosion then dilation: removes small specks.", morph_params, _morph(cv2.MORPH_OPEN), output="mask"),
        Operation("close", 2, "Morphology", "Closing", "Dilation then erosion: fills small gaps.", morph_params, _morph(cv2.MORPH_CLOSE), output="mask"),
        Operation("plate_shapes", 2, "Shape-based", "Plate-shaped regions", "Keeps wide, solid blobs of vertical edges (plate candidates).",
                  (num("min_aspect", "Min aspect", 2, 1, 10, 0.1), num("max_aspect", "Max aspect", 6, 1.5, 12, 0.1)), _plate_shapes, output="mask"),
        # Module 3
        Operation("translate", 3, "Transformations", "Translation", "Shifts the image.",
                  (num("tx", "Shift x (px)", 40, -2000, 2000), num("ty", "Shift y (px)", 20, -2000, 2000), INTERP_PARAM, MAPPING_PARAM), _translate),
        Operation("scale", 3, "Transformations", "Scaling (zoom)",
                  "Enlarges the image, or in zoom-in mode magnifies around a point and keeps the frame size.",
                  (num("factor", "Factor", 1.5, 0.1, 4, 0.05),
                   sel("mode", "Mode", "enlarge", ["enlarge", "zoom-in"]),
                   num("center_x", "Zoom-in centre x (%)", 50, 0, 100, 1),
                   num("center_y", "Zoom-in centre y (%)", 50, 0, 100, 1),
                   INTERP_PARAM), _scale),
        Operation("rotate", 3, "Transformations", "Rotation", "Rotates counter-clockwise about the centre.",
                  (num("angle", "Angle (deg)", 10, -180, 180, 0.5), sel("enlarge", "Enlarge canvas", "no", ["no", "yes"]), INTERP_PARAM, MAPPING_PARAM), _rotate),
        Operation("shear", 3, "Transformations", "Shear (affine)", "General affine example: slants rows or columns.",
                  (num("shear_x", "Horizontal shear", 0.2, -1, 1, 0.01), num("shear_y", "Vertical shear", 0, -1, 1, 0.01), INTERP_PARAM), _shear),
        Operation("radial", 3, "Lens correction", "Radial distortion correction", "Polynomial lens model. Coefficients must come from calibration.",
                  (num("k1", "k1", 0.1, -1, 1, 0.01), num("k2", "k2", 0, -1, 1, 0.01), INTERP_PARAM), _radial),
        Operation("register", 3, "Registration", "Image registration (ORB + RANSAC)", "Aligns the image to a reference image. OpenCV only.",
                  (INTERP_PARAM,), _register, needs_reference=True, matlab=False),
    ]


CATALOG: dict[str, Operation] = {op.id: op for op in _ops()}


def resolve_params(op: Operation, given: Optional[dict]) -> dict:
    """Fill defaults and validate values against the catalog. Raises InvalidParamsError."""
    given = given or {}
    out: dict = {}
    for prm in op.params:
        val = given.get(prm.key, prm.default)
        if prm.kind == "select":
            if val not in prm.options:
                raise InvalidParamsError(f"{op.label}: '{prm.label}' must be one of {', '.join(prm.options)}.")
        else:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val):
                raise InvalidParamsError(f"{op.label}: '{prm.label}' must be a number.")
            if not (prm.min <= val <= prm.max):
                raise InvalidParamsError(f"{op.label}: '{prm.label}' must be between {prm.min:g} and {prm.max:g}.")
        out[prm.key] = val
    if op.id == "fourier_band" and out["outer"] <= out["inner"]:
        raise InvalidParamsError("Fourier band-pass: the outer radius must be larger than the inner radius.")
    if op.id == "canny" and out["high"] <= out["low"]:
        raise InvalidParamsError("Canny: the high threshold must be larger than the low threshold.")
    if op.id == "plate_shapes" and out["max_aspect"] <= out["min_aspect"]:
        raise InvalidParamsError("Plate shapes: max aspect must be larger than min aspect.")
    return out


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
