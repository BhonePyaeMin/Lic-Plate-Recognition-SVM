# Smart License Plate Detection and Recognition System

A web application that locates a vehicle license plate in an uploaded photo and reads its
characters. A React frontend talks to a FastAPI backend that runs an OpenCV detection
pipeline and an OCR engine. The repository also holds the MATLAB image-processing toolbox
and GUI required by the course brief.

> **Course alignment.** The assigned topic is "License Plate Recognition System Using OpenCV
> in Python". The brief also requires a MATLAB toolbox, a MATLAB GUI and three modules. The
> web app can run every filter in either OpenCV (always available) or MATLAB (when MATLAB and its
> Python Engine are installed on the server). **MATLAB was not installed where this was built,
> so the MATLAB path has never run**; until you install it, the header shows "MATLAB: not
> installed" and OpenCV does the work. See [MATLAB integration](#matlab-integration).

## Project status

| Area | Status |
|---|---|
| FastAPI backend, upload validation, crop and filter API | Implemented, 98 automated tests pass |
| 29 filters across Modules 1 to 3 (OpenCV/NumPy) | Implemented; each runs with defaults in tests |
| OpenCV baseline detector | Implemented; verified on synthetic scenes only |
| OCR (RapidOCR / ONNX Runtime) | Implemented; verified on synthetic scenes only |
| React frontend (crop, filter studio, compare, results) | Implemented, 24 automated tests pass, checked in a browser |
| MATLAB Modules 1 to 3 and GUI | Written, **never executed** (no MATLAB available when written) |
| MATLAB Engine bridge (`run_operation_file.m` + adapter) | Written and wired into the pipeline; engine routing and fallback are tested with a fake, the real MATLAB call is **untested** |
| Image registration | Implemented with OpenCV only (ORB + RANSAC); no MATLAB version (TODO) |
| Backend Dockerfile | Written, **not built** (Docker daemon unavailable) |
| Accuracy on real photographs | **Not measured.** No real images or labels are included |

## Features

* Upload by drag and drop or file browser; formats and size limit come from the backend `/health`.
* **Prepare stage before detection:** draw a crop rectangle on the image (or type exact pixels),
  add up to 8 filters in order, adjust every parameter, preview the processed image next to the
  original, then detect. The detector and OCR run on the cropped and filtered image.
* **All course filters are usable:** histogram equalization and matching, brightness/contrast,
  gamma, mean and median smoothing, Laplacian and high-boost sharpening, Fourier low/high/band-pass;
  Sobel, Prewitt, Canny, Otsu, region growing, RGB and HSV segmentation, dilation, erosion, opening,
  closing, plate-shape filtering; translation, scaling, rotation, shear, radial distortion
  correction (forward and reverse mapping, nearest/bilinear/bicubic interpolation); image
  registration. Segmentation filters output a black and white mask, so they are meant for
  inspection and comparison; text reading generally works better on a normal image.
* **Compare methods:** run all edge detectors, three segmentation methods or the four
  morphological operations side by side on the current crop.
* **Engine choice:** Automatic (MATLAB when ready, else OpenCV), OpenCV only, or MATLAB only. The
  preview and the results show which engine ran each step.
* Presets for low light, noisy and soft-focus photos.
* If the crop is tight around the plate and the detector finds no text, the whole selected area is read and reported
  as detector `user-crop`.
* Processing starts only when the user presses **Detect plate**.
* Result view: plate text with copy action, OCR confidence (only when the engine returns one),
  original image, annotated image, plate crop, server-measured processing time and metadata.
* Skeleton loading state that mirrors the result layout, with reduced-motion support.
* Explicit states for no plate found, unreadable plate, unsupported type, oversized file,
  corrupted image, API unavailable and processing failure.
* DM Sans (self-hosted through `@fontsource-variable/dm-sans`), Lucide icons, white theme,
  responsive from phone to desktop.

## Stack

React 18 + Vite, Python 3.11, FastAPI, OpenCV (headless), RapidOCR (ONNX Runtime), MATLAB
Image Processing Toolbox (separate).

Why RapidOCR: it installs with `pip` alone and bundles its models, so no system binary
(Tesseract) or first-run model download (EasyOCR) is needed. OCR sits behind the
`PlateRecognizer` interface in `backend/app/services/plate_recognizer.py`; replace it to use
EasyOCR or Tesseract. Its bundled model reads Chinese and English; output is restricted to
A-Z and 0-9.

## Repository layout

```text
backend/    FastAPI app (app/), tests, requirements, Dockerfile
frontend/   React + Vite app (src/), tests
matlab/     module1_enhancement, module2_segmentation, module3_transformations, toolbox_gui, tests
samples/    generator for synthetic demo images
docs/       architecture.md, deployment.md
.env.example
```

## Local setup

Requirements: Python 3.10 to 3.12 (3.11 was used), Node.js LTS.

### Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: <http://127.0.0.1:8000/health>. Interactive docs: <http://127.0.0.1:8000/docs>.

Configuration is read from environment variables; see `.env.example` (the backend does not
load `.env` files itself, export the variables or set them in your host).

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local     # set VITE_API_BASE_URL if the backend is not on 127.0.0.1:8000
npm run dev
```

Open the URL Vite prints (default <http://localhost:5173>).

### Demo images

```bash
backend/.venv/Scripts/python samples/generate_samples.py   # macOS/Linux: backend/.venv/bin/python
```

This writes four drawn images (including one with no plate) into `samples/`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status, OCR and MATLAB capability, upload limits |
| GET | `/api/v1/filters` | Catalog of operations with parameter ranges and defaults |
| POST | `/api/v1/filters/apply` | Crop and filter only (preview); returns the processed image |
| POST | `/api/v1/detect` | Detect and read a plate, optionally after crop and filters |

Form fields for both POST endpoints: `file` (required), `crop` (JSON `{"x","y","width","height"}` in
original pixels), `steps` (JSON list `[{"id": "median", "params": {"kernel": 5}}]`, at most 8),
`engine` (`auto`, `opencv` or `matlab`), `reference` (image, needed by histogram matching and
registration). Invalid crop, steps or parameters return `400 INVALID_PARAMS`; `engine=matlab`
without MATLAB returns `503 MATLAB_UNAVAILABLE`. In the result, `bounding_box` and `image_*`
are in the coordinates of the cropped and filtered image, `input_image` is that image, and
`preprocessing` lists the crop and each step with the engine that ran it.

Success (`200`), shape only (every field is computed per request, none of these are real values):

```json
{
  "success": true,
  "message": "License plate processed",
  "result": {
    "detection_status": "recognized",
    "plate_text": "<string>",
    "raw_ocr_text": "<string>",
    "confidence": null,
    "bounding_box": { "x": 0, "y": 0, "width": 0, "height": 0 },
    "detector": "opencv-contour-baseline",
    "detector_score": 0.0,
    "ocr_engine": "rapidocr-onnxruntime",
    "image_width": 0,
    "image_height": 0,
    "processing_time_ms": 0.0,
    "annotated_image": "data:image/jpeg;base64,...",
    "plate_crop": "data:image/png;base64,...",
    "matlab_used": false
  }
}
```

`confidence` is `null` when the engine reports none. `detection_status` is `located_unreadable`
(with `plate_text: null`) when a region was found but OCR returned nothing usable.

Errors share one shape, `{"success": false, "message": "...", "error_code": "..."}`:

| Situation | HTTP | `error_code` |
|---|---|---|
| Empty upload | 400 | `EMPTY_FILE` |
| Corrupt image, too many pixels | 400 | `INVALID_IMAGE` |
| Missing `file` field | 400 | `INVALID_REQUEST` |
| Unsupported type or extension/content mismatch | 415 | `UNSUPPORTED_FILE_TYPE` |
| Larger than `MAX_UPLOAD_MB` | 413 | `FILE_TOO_LARGE` |
| No plate region found | 200 | `PLATE_NOT_FOUND` |
| OCR engine failed to load | 503 | `OCR_UNAVAILABLE` |
| Unexpected failure (no stack trace returned) | 500 | `PROCESSING_FAILED` |

Validation combines size, emptiness, extension, MIME type (can only veto), a Pillow signature
check, a pixel-count cap and a full OpenCV decode. Images are processed in memory; the client
filename is never used as a path and nothing is written to disk.

## Detection pipeline and limitations

Details are in [`docs/architecture.md`](docs/architecture.md). In short: grayscale, bilateral
denoise, black-hat and top-hat to isolate character strokes, a horizontal gradient closed into
blobs, contour filtering by aspect ratio and area, ranking by a documented weighted score, OCR on
the top candidates, deskew of the chosen crop, conservative normalisation (upper case, A-Z0-9 only,
no O/0 substitution).

Known limits of the baseline:

* Single-row plates only (aspect 1.8 to 8); two-row plates are missed.
* Blur, glare, strong perspective and very low resolution degrade it.
* Textured backgrounds (grilles, fences, signs) can produce false regions; a "possible plate
  region, text unreadable" result can be a false positive.
* The detector score is a ranking heuristic, not a probability.
* The bounding box hugs the characters plus a fixed margin, so it can be smaller than the physical plate.
* Synthetic scenes were the only test material; real-world accuracy is unknown.
* CPU inference took roughly 0.5 to 2 s per synthetic image on the development machine (the first
  request includes model loading).

## Testing

```bash
cd backend && python -m pytest          # 98 tests
cd frontend && npm test                 # 24 tests (Vitest + Testing Library)
```

Results recorded when this was written: backend `98 passed`, frontend `24 passed`. Backend API
tests use a fake OCR so they do not depend on model files; one test uses the real OCR engine and
skips itself if it cannot load. The frontend "layout" test only checks that the stylesheet
contains the expected breakpoints and reduced-motion rules; the actual responsive layout was
checked by hand in a browser at desktop width and 375 px width.

MATLAB: run `matlab/tests/run_all_tests.m` in MATLAB (not yet executed by the author).

## MATLAB integration

* **Runtime integration (built in, on by default).** At start-up the backend looks for MATLAB
  Engine for Python. If it is installed, MATLAB starts in the background and every filter except
  registration then runs through `matlab/api/run_operation_file.m` (same operation ids and
  parameters as the OpenCV versions; images are exchanged as temporary PNG files that are deleted
  after each call). The header shows "MATLAB: starting" then "MATLAB: ready". If a MATLAB call
  fails in Automatic mode, OpenCV runs that step instead and the UI says so.
* **To turn it on:** install MATLAB with the Image Processing Toolbox on the machine that runs the
  backend, then in the backend virtual environment run `pip install matlabengine` (use the version
  that matches your MATLAB release; see MathWorks documentation), and restart the backend. No other
  setting is needed (`ENABLE_MATLAB` defaults to true; `MATLAB_PATH` overrides the `matlab/` folder).
* **Not verified:** none of this has run against real MATLAB. Run `matlab/tests/run_all_tests.m`
  first, then try one filter with engine "MATLAB only" and fix what the first run reports.
* **Separate demonstration:** the standalone GUI (`matlab/toolbox_gui/plate_toolbox_gui.m`) works
  without Python.

Cloud hosts normally lack MATLAB, so runtime integration is realistic only on a machine you control.

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for exact steps (static frontend plus containerised
API, CORS, verification checklist).

## Privacy

Uploads are decoded in memory and discarded after the response. Images are not logged or stored.
Only process images you have permission to use.

## License

Add the license chosen by your team before public distribution.
