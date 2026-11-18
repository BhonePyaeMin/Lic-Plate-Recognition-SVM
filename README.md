# Smart License Plate Detection and Recognition System

A web application that locates a vehicle license plate in an uploaded photo and reads its
characters. A React frontend talks to a FastAPI backend that runs an OpenCV detection
pipeline and an OCR engine. The repository also holds the MATLAB image-processing toolbox
and GUI required by the course brief.

> **Course alignment.** The assigned topic is "License Plate Recognition System Using OpenCV
> in Python". The brief also requires a MATLAB toolbox, a MATLAB GUI and three modules. The
> web app can run every filter in either OpenCV (always available) or MATLAB (when MATLAB and its
> Python Engine are installed on the server). **MATLAB was not installed where this was first built.
> It has since been run on MATLAB R2026a** (module tests and every filter through the Engine, see
> below); without MATLAB the header shows "MATLAB: not installed" and OpenCV does the work. See [MATLAB integration](#matlab-integration).

## Project status

| Area | Status |
|---|---|
| FastAPI backend, upload validation, crop and filter API | Implemented, 150 automated tests pass |
| 29 filters across Modules 1 to 3 (OpenCV/NumPy) | Implemented; each runs with defaults in tests |
| OpenCV baseline detector | Implemented; verified on synthetic scenes only |
| OCR (RapidOCR / ONNX Runtime) | Implemented; verified on synthetic scenes only |
| SVM plate verification (HOG + linear SVM) | Implemented and trained on **synthetic scenes only**; see [SVM plate verification](#svm-plate-verification) |
| SVM character recognition (second OCR engine) | Implemented and trained on **synthetic plates only**; see [SVM character recognition](#svm-character-recognition) |
| React frontend (crop, filter studio, compare, results) | Implemented, 42 automated tests pass; the new layout, both SVM selectors and the SVM recognition panel were checked in a browser |
| MATLAB Modules 1 to 3 | Run on MATLAB R2026a, 27 checks pass; need no toolbox (`matlab/common`). The standalone GUI is **not yet run** |
| MATLAB Engine bridge (`run_operation_file.m` + adapter) | Written and wired into the pipeline; engine routing and fallback are tested with a fake; all 28 MATLAB-capable filters and one detection were run through a real MATLAB R2026a Engine session (Windows, Python 3.12) |
| Image registration | Implemented with OpenCV only (ORB + RANSAC); no MATLAB version (TODO) |
| Backend Dockerfile | Built and run on Docker Desktop (Windows) before the SVM was added; it needed three extra system libraries, now in the file. The `COPY models` line for the SVM was added afterwards and has **not been rebuilt** |
| Accuracy on real photographs | **Not measured**, for the heuristic detector, the SVM and OCR alike. No real images or labels are included |

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

React 18 + Vite, Python 3.11, FastAPI, OpenCV (headless), RapidOCR (ONNX Runtime), a HOG + linear
SVM written in numpy (trained with scikit-learn, not needed to serve), MATLAB (base MATLAB only;
the Image Processing Toolbox is not required).

Why RapidOCR: it installs with `pip` alone and bundles its models, so no system binary
(Tesseract) or first-run model download (EasyOCR) is needed. OCR sits behind the
`PlateRecognizer` interface in `backend/app/services/plate_recognizer.py`; replace it to use
EasyOCR or Tesseract. Its bundled model reads Chinese and English; output is restricted to
A-Z and 0-9.

## Repository layout

```text
backend/    FastAPI app (app/), tests, requirements, Dockerfile, models/ (SVM weights), scripts/ (SVM training)
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
| GET | `/health` | Service status, OCR, SVM (plate and character) and MATLAB capability, upload limits |
| GET | `/api/v1/filters` | Catalog of operations with parameter ranges and defaults |
| POST | `/api/v1/filters/apply` | Crop and filter only (preview); returns the processed image |
| POST | `/api/v1/detect` | Detect and read a plate, optionally after crop and filters |

Form fields for both POST endpoints: `file` (required), `crop` (JSON `{"x","y","width","height"}` in
original pixels), `steps` (JSON list `[{"id": "median", "params": {"kernel": 5}}]`, at most 8),
`engine` (`auto`, `opencv` or `matlab`), `reference` (image, needed by histogram matching and
registration). `/detect` also takes `detector_mode` (`auto`, `contour` or `svm`, default `auto`; see
[SVM plate verification](#svm-plate-verification)) and `ocr_engine` (`auto`, `rapidocr` or `svm`, default
`auto`; see [SVM character recognition](#svm-character-recognition)). Invalid crop, steps or parameters return `400 INVALID_PARAMS`; `engine=matlab`
without MATLAB returns `503 MATLAB_UNAVAILABLE`; `detector_mode=svm` without a model returns
`503 SVM_UNAVAILABLE`, and `ocr_engine=svm` without a model returns `503 SVM_OCR_UNAVAILABLE`. In the result, `bounding_box` and `image_*`
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
| `detector_mode=svm` but no SVM model is loaded | 503 | `SVM_UNAVAILABLE` |
| `ocr_engine=svm` but no character model is loaded | 503 | `SVM_OCR_UNAVAILABLE` |
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

## SVM plate verification

A Support Vector Machine gives the OpenCV detector a learned second opinion. The OpenCV baseline
still proposes candidate regions; each candidate crop is resized to 128x32, described with
Histogram of Oriented Gradients (HOG) features (1620 values) and scored by a linear SVM trained
to separate "plate" from "not a plate". Code: `backend/app/services/svm_plate.py` (HOG written in
numpy, so serving needs only numpy and OpenCV) and `backend/scripts/train_svm.py`.

The `detector_mode` field of `/api/v1/detect` (and the "Plate detection method" menu in the UI):

| Mode | Behaviour |
|---|---|
| `contour` | Heuristic detector only (the original behaviour). |
| `auto` (default) | The SVM **re-ranks** candidates and never drops one. Falls back to `contour` when no model is loaded. |
| `svm` | The SVM re-ranks and also **rejects** candidates whose margin is below the model's threshold. `503 SVM_UNAVAILABLE` without a model. |

The result carries `svm_used` and `svm_score`. The score is the SVM margin: positive means the
plate side of the decision boundary, larger means further from it. It is **not** a probability.

**Training data is synthetic.** There are no labelled real photographs in this repository, so the
model in `backend/models/` was trained on drawn scenes (random plates, fonts, tilt, noise, plus
grilles, fences and stripes as clutter). Candidates the baseline detector produced on those scenes
were labelled by overlap with the true plate, so the negatives are the detector's own false alarms.
Numbers on **held-out synthetic scenes** (seeds not used in training; 300 test scenes):

| Measure | Result |
|---|---|
| Candidate classification, ROC AUC | 0.9993 |
| Candidate classification, precision / recall | 0.9615 / 0.9983 |
| Top-1 region correct (plate scenes), heuristic only | 60% |
| Top-1 region correct (plate scenes), SVM in `svm` mode | 67% |
| Scenes without a plate that still return a region, heuristic only | 73% |
| Scenes without a plate that still return a region, `svm` mode | 7% |

The SVM cannot find plates the OpenCV proposer misses (about 30% of plate scenes on this
synthetic set), so recall is bounded by the proposer. These figures say nothing about real photographs: the
drawn plates are far cleaner than real ones. `auto` mode never rejects a region for that reason;
use `svm` mode only after checking it on your own images.

Retrain (needs `pip install -r requirements-train.txt`, a few minutes) and add real labelled
images to the generator or dataset code before relying on it outside demonstrations:

```bash
cd backend
python scripts/train_svm.py            # writes models/plate_svm.npz and plate_svm.json
```

Environment: `ENABLE_SVM`, `SVM_MODEL_PATH`, `SVM_MIN_MARGIN` (see `.env.example`).

## SVM character recognition

A second, selectable OCR engine built from classical parts instead of a neural network. The plate
image is binarised (both ink polarities are tried), connected components are filtered to
character-sized shapes, clearly merged pairs are split at a deep valley of the ink projection, and
each glyph is fitted to a 24x48 canvas, described with HOG features plus a thumbnail and its aspect
ratio (433 values), and classified by a 36-class SVM (0-9, A-Z). The shipped model is an RBF-kernel
SVM approximated with 1500 random Fourier features (a linear SVM on top), trained with hinge loss. Code:
`backend/app/services/svm_ocr.py`; training: `backend/scripts/train_svm_ocr.py`. Serving needs only
numpy and OpenCV; the model file is about 2.6 MB.

Choose it with the `ocr_engine` field of `/api/v1/detect` or the "Text recognition" menu in the UI:

| Value | Behaviour |
|---|---|
| `auto` (default) | The SVM reads first and its text is the answer. RapidOCR reads the same plate as a **cross-check**: the result carries `ocr_alternative`, and the UI shows "RapidOCR agrees" or what RapidOCR read instead. If the SVM finds fewer than 4 letters or digits, RapidOCR reads instead and the result is flagged `ocr_fallback: true`. Without a character model it is simply RapidOCR. |
| `svm` | SVM only, no fallback and no cross-check. `503 SVM_OCR_UNAVAILABLE` when no model is loaded. |
| `rapidocr` | RapidOCR only. |

The result lists each character with its SVM margin in `svm_chars`; no `confidence` is reported
because a margin is not a probability.

**SVM recognition panel.** Every detection also returns `candidates`: each candidate plate that was
read, with its status (`readable`, `small` or `unread`), crop, text, and for the SVM engine the plate
image with the characters it segmented boxed and the normalised glyphs it classified. The UI shows
them in an "SVM recognition" card: candidates are numbered on the image (red = readable,
orange = small, grey = not read) and clicking one shows its segmentation and glyphs.

**Training data is synthetic.** Plates are drawn with four OpenCV fonts plus 51 TrueType
faces found on the training machine (sans, gothic, DIN-style, condensed, light, monospaced and
OCR-style), in four colour schemes, with tilt, perspective, blur, resolution loss, gamma and uneven
lighting, JPEG compression, and real-world traits such as a blue country strip with small letters and
a line of tiny print under the characters. They then pass through the same deskew, preparation and
segmentation code the server uses, and every training glyph is also augmented three times (heavier
or lighter strokes, rotation, shear, width). No font file is distributed; only the learned weights
are. The test plates use only 16 typefaces held out of training
(arialbd, brlnsr, candara, consolab, cour, dubai-medium, eraslght, fradmcn, gadugi, gilsanub, lsans, malgun, ocraext, segoeuil, tahomabd, trebuc). Numbers on **800 held-out synthetic plates in unseen
typefaces**, which are deliberately harder than a clean drawing:

| Measure | Result |
|---|---|
| Plates segmented into the right number of characters | 86.8% |
| Character accuracy on correctly segmented plates | 96.5% |
| Whole plate exactly right (segmentation failures count as wrong) | 74.1% |
| RapidOCR, whole plate exactly right, 150 of the same synthetic plates | 63.3% |

The most common character mistakes are the look-alike pairs (O->0, 0->O, 1->I, 8->6, 2->E); no context
rules repair them.

**A real-photograph check, and what it showed.** An earlier model, trained on four drawn fonts only,
read a real photographed plate (a Romanian plate, `IF 12 YCC`, on a night shot of a car) as `IFI2BCC`:
it turned `1` into `I` and a clear `Y` into `B`. That is the synthetic-to-real gap in action, and it
is why the training data above was widened. The current model reads a crop of that same plate
correctly (`IF12YCC`) at three sizes down to about 200 px wide, and slips (`IY12YCC`) at about 150 px.
This is one plate, cropped from a screenshot and never used for training, so it shows the direction
of the fix, not an accuracy figure. Real-photograph accuracy has not been measured. RapidOCR, on that
same crop, read the blue country strip as an `L` (`LTF12YCC`), so neither engine is always right,
which is why Automatic mode shows the cross-check instead of hiding it.

Dirt, glare, heavy blur and characters touching the frame have not been tested. Check it on your own
images, and retrain on real labelled plates before relying on it:

```bash
cd backend
python scripts/train_svm_ocr.py        # writes models/ocr_svm.npz and ocr_svm.json (about 5 minutes)
```

Environment: `ENABLE_SVM_OCR`, `SVM_OCR_MODEL_PATH` (see `.env.example`).

## Testing

```bash
cd backend && python -m pytest          # 150 tests
cd frontend && npm test                 # 42 tests (Vitest + Testing Library)
```

Results recorded when this was written: backend `150 passed`, frontend `42 passed`. Backend API
tests use a fake OCR and a disabled or stub SVM so they do not depend on model files; one test uses the real OCR engine and
skips itself if it cannot load. The frontend "layout" test only checks that the stylesheet
contains the expected breakpoints and reduced-motion rules; the actual responsive layout was
checked by hand in a browser at desktop width and 375 px width.

MATLAB: run `matlab/tests/run_all_tests.m` in MATLAB (27 checks, all passed on R2026a).

## MATLAB integration

* **Runtime integration (built in, on by default).** At start-up the backend looks for MATLAB
  Engine for Python. If it is installed, MATLAB starts in the background and every filter except
  registration then runs through `matlab/api/run_operation_file.m` (same operation ids and
  parameters as the OpenCV versions; images are exchanged as temporary PNG files that are deleted
  after each call). The header shows "MATLAB: starting" then "MATLAB: ready". If a MATLAB call
  fails in Automatic mode, OpenCV runs that step instead and the UI says so.
* **To turn it on:** install MATLAB (no toolbox needed) on the machine that runs the
  backend, then in the backend virtual environment run `pip install matlabengine` (use the version
  that matches your MATLAB release, and a Python version that release supports; R2026a supports
  Python 3.9 to 3.13, while `rapidocr-onnxruntime` needs 3.12 or older, so use 3.12), and restart the backend. No other
  setting is needed (`ENABLE_MATLAB` defaults to true; `MATLAB_PATH` overrides the `matlab/` folder).
* **Verified so far:** on MATLAB R2026a every filter runs with engine "MATLAB only" and no fallback.
  Some outputs differ slightly from OpenCV (for example gamma, and the rectangle structuring element
  height: MATLAB rounds 2.5 up to 3, Python rounds it to 2). Other MATLAB releases were not tried.
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

MIT. See [LICENSE](LICENSE).
