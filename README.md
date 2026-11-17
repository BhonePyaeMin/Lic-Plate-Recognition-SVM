# Smart License Plate Detection and Recognition System

A web-based application for detecting vehicle license plates and
recognizing their characters from uploaded images.

> **Course alignment:** The assigned topic is "License Plate Recognition
> System Using OpenCV in Python." The project brief also requires a
> MATLAB-based image-processing toolbox, a MATLAB GUI, and three
> image-processing modules. This repository is designed to support a
> Python web application alongside the required MATLAB coursework.
> Confirm the expected level of runtime integration with your lecturer.

## Features

### Planned application features

-   Upload a vehicle image using file selection or drag and drop.
-   Detect a license plate and locate it with a bounding box.
-   Crop and preprocess the detected plate.
-   Recognize plate characters using an OCR engine.
-   Display the original image, processed image, plate crop, and
    recognition result.
-   Provide clear loading, empty, success, no-detection, and error
    states.
-   Use a responsive interface with DM Sans typography and consistent
    icons.
-   Use skeleton animations during loading and processing.

### Course image-processing modules

  -----------------------------------------------------------------------
  Module                              Scope
  ----------------------------------- -----------------------------------
  Module 1: Image Enhancement         Histogram processing,
                                      brightness/contrast, spatial
                                      filters, and frequency-domain
                                      filtering

  Module 2: Segmentation and Edge     Sobel, Prewitt, Canny,
  Detection                           thresholding, region-based
                                      segmentation, color segmentation,
                                      and morphology

  Module 3: Geometric Transformations Translation, scaling, rotation,
  and Interpolation                   affine transformations, mapping,
                                      distortion correction, and
                                      interpolation
  -----------------------------------------------------------------------

These are assignment requirements to implement and demonstrate. Do not
describe a technique as completed until its implementation and test are
available.

## Technology Stack

-   **Frontend:** React + Vite
-   **Backend:** Python + FastAPI
-   **Computer vision:** OpenCV
-   **OCR:** EasyOCR or Tesseract
-   **Course toolbox:** MATLAB Image Processing Toolbox
-   **Typography:** DM Sans
-   **Icons:** Lucide
-   **Optional database:** SQLite

## Architecture

``` text
User
 |
 v
React Web Application
 |
 | HTTP / REST
 v
FastAPI Backend
 |
 +--> Upload validation
 |
 +--> OpenCV preprocessing and plate detection
 |
 +--> Plate crop and image preparation
 |
 +--> OCR recognition
 |
 +--> Optional MATLAB adapter
 |
 v
Structured detection result
 |
 v
Web result visualization
```

See [`docs/architecture.md`](docs/architecture.md) for the full
architecture, processing flow, API schema, MATLAB mapping, testing, and
deployment plan.

## Project Structure

``` text
license-plate-recognition/
├── frontend/       # React web application
├── backend/        # FastAPI and computer-vision services
├── matlab/         # Three required MATLAB modules and GUI
├── samples/        # Approved sample images for testing
├── docs/           # Architecture and project documentation
├── .env.example
├── .gitignore
└── README.md
```

The structure is a recommended starting point. Update it to match the
actual files as implementation progresses.

## Requirements

-   Python 3.10 or newer (use a version supported by the selected OpenCV
    and OCR packages).
-   Node.js LTS and npm.
-   Git.
-   MATLAB and the required Image Processing Toolbox for the MATLAB
    coursework.
-   A working camera or sample vehicle images for testing.

## Local Setup

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd license-plate-recognition
```

### 2. Start the backend

``` bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

``` powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

``` cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start FastAPI:

``` bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

-   API health: `http://127.0.0.1:8000/health`
-   Interactive API documentation: `http://127.0.0.1:8000/docs`

### 3. Start the frontend

Open a second terminal:

``` bash
cd frontend
npm install
```

Create a local environment file based on `.env.example` and configure
the backend URL. For example:

``` env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Run the frontend:

``` bash
npm run dev
```

Open the local URL printed by Vite in the terminal.

## API Overview

Base path: `/api/v1`

  Method   Endpoint                 Description
  -------- ------------------------ -------------------------------------------------
  GET      `/health`                Backend health check
  POST     `/detect`                Upload an image and run plate detection and OCR
  GET      `/history`               Optional processing history
  GET      `/history/{result_id}`   Optional single history record

### Detection request

`POST /api/v1/detect`

Content type: `multipart/form-data`

  Field    Type         Description
  -------- ------------ --------------------------
  `file`   Image file   Vehicle image to process

### Example success response

``` json
{
  "success": true,
  "message": "License plate processed",
  "result": {
    "plate_text": "ABC1234",
    "raw_ocr_text": "ABC 1234",
    "confidence": 0.91,
    "bounding_box": {
      "x": 120,
      "y": 210,
      "width": 260,
      "height": 80
    },
    "image_width": 1280,
    "image_height": 720,
    "processing_time_ms": 245
  }
}
```

This is an example contract only. Actual fields must reflect the
implemented detector and OCR engine. Never fabricate confidence or
processing-time values.

### Example no-detection response

``` json
{
  "success": false,
  "message": "No license plate detected",
  "error_code": "PLATE_NOT_FOUND"
}
```

## MATLAB Assignment Alignment

The assignment's three modules should be implemented and documented in
MATLAB:

1.  **Image Enhancement:** histogram equalization and matching,
    brightness/contrast transformations, mean and median smoothing,
    Laplacian/high-pass sharpening, and Fourier low-pass, high-pass, and
    band-pass filtering.
2.  **Image Segmentation and Edge Detection:** Sobel, Prewitt, Canny,
    Otsu thresholding, region-based methods, RGB/HSV segmentation,
    dilation, erosion, opening, closing, and shape-based segmentation.
3.  **Geometric Transformations and Interpolation:** translation,
    scaling, rotation, affine transformations, forward/reverse mapping,
    radial distortion correction, and nearest-neighbor, bilinear, and
    bicubic interpolation. Image registration is optional.

The final project brief also calls for an integrated MATLAB GUI that
allows users to upload an image, choose a processing technique, and
visualize results.

### MATLAB and Python integration

Two valid implementation paths are proposed, subject to lecturer
approval:

-   **Separate demonstrations:** demonstrate the MATLAB toolbox/GUI and
    the Python web-based license-plate recognition system separately,
    using common sample images.
-   **Runtime integration:** call selected MATLAB functions from the
    Python backend through MATLAB Engine for Python, if the environment
    and license support it.

Do not assume MATLAB Engine is available on a cloud deployment host.
Document the chosen approach clearly.

## Interface Design Requirements

-   Use DM Sans for headings, body text, labels, and controls.
-   Use icons from Lucide or a consistent icon library. Do not use emoji
    as icons.
-   Create a polished, responsive interface with a clear visual
    hierarchy.
-   Use a dedicated upload area with drag-and-drop and browse actions.
-   Present original image, detected image, plate crop, and recognized
    text clearly.
-   Add skeleton loaders that match the dimensions and structure of the
    result components.
-   Include empty, processing, success, no-result, and error states.
-   Respect reduced-motion preferences.
-   Provide accessible labels, keyboard navigation, visible focus
    states, and sufficient color contrast.

## Testing Checklist

-   [ ] Valid JPG, JPEG, and PNG uploads work.
-   [ ] Unsupported and corrupted files are rejected safely.
-   [ ] Oversized uploads are rejected with a clear message.
-   [ ] A clear, front-facing plate can be detected.
-   [ ] Tilted and rotated plates are tested.
-   [ ] Low-light, overexposed, blurred, and noisy images are tested.
-   [ ] Images without a plate return a no-detection result.
-   [ ] OCR output is compared against manually labeled ground truth.
-   [ ] Bounding boxes align with the plate.
-   [ ] Loading skeletons and error states render correctly.
-   [ ] MATLAB module functions are individually tested.
-   [ ] The MATLAB GUI demonstrates the required toolbox functions.
-   [ ] The deployed frontend can reach the deployed backend.

## Deployment

Recommended deployment:

1.  Build and deploy the React frontend to a static hosting provider.
2.  Deploy FastAPI to a Python-capable host or Docker container
    platform.
3.  Configure the frontend API base URL using an environment variable.
4.  Configure production CORS to allow only the deployed frontend
    origin.
5.  Verify the backend health endpoint and test an end-to-end image
    upload.
6.  Ensure temporary image files are not exposed as public assets.

MATLAB runtime integration requires a compatible deployment environment
and licensing. If that is not available, keep the MATLAB GUI/toolbox as
a separate course demonstration and state this clearly in the report.

## Privacy and Safety

-   Process only images you have permission to use.
-   Avoid uploading images containing sensitive information unless
    necessary and authorized.
-   Validate uploaded files and generate server-side filenames.
-   Keep temporary uploads outside public directories and delete them
    after processing.
-   Do not commit secrets, private images, or unneeded test data.
-   Do not expose server stack traces to users.

## Project Status

This README describes the planned system and requirements. Update this
section as features are implemented and verified.

  Area                     Status
  ------------------------ ---------
  Frontend                 Planned
  FastAPI backend          Planned
  OpenCV plate detection   Planned
  OCR integration          Planned
  MATLAB Module 1          Planned
  MATLAB Module 2          Planned
  MATLAB Module 3          Planned
  MATLAB GUI               Planned
  Deployment               Planned

## License

Add the license selected by your project team before public
distribution.
