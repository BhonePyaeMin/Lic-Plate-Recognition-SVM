# Claude Code Implementation and Deployment Prompt

## Smart License Plate Detection and Recognition System

Act as a senior full-stack engineer, computer-vision engineer, and UI/UX
designer. Build this project carefully in the current repository. First
inspect the existing files and preserve working code. Do not overwrite
or remove existing work without a clear reason.

## 1. Project Goal

Build a polished web application for **License Plate Detection and
Recognition**.

The assigned project topic is "License Plate Recognition System Using
OpenCV in Python." The course brief additionally requires a MATLAB-based
image-processing toolbox, a MATLAB GUI, and three modules:

-   Module 1: Image enhancement and filtering.
-   Module 2: Segmentation, edge detection, and morphological
    operations.
-   Module 3: Geometric transformations and interpolation.

Use Python FastAPI as the web API, OpenCV for the Python computer-vision
pipeline, and an OCR engine such as EasyOCR or Tesseract. Keep MATLAB
functions in a separate `matlab/` area and design a clean optional
adapter. Do not pretend MATLAB is integrated if it is not. Do not make
the entire project Python-only while claiming the MATLAB requirements
are fulfilled.

Before implementation, inspect the repository and identify the current
state. Then create a concise implementation plan and execute it in
logical stages. If the repository is empty, scaffold it using the
structure below.

## 2. Mandatory Design Rules

These requirements are strict:

-   Do not use emoji anywhere in the application, documentation UI
    examples, or interface labels.
-   Use icons, preferably Lucide icons, consistently. Do not use emoji
    as substitutes for icons.
-   Use **DM Sans** as the primary font across the entire frontend.
-   Create a fantastic, professional, modern interface. Prioritize
    excellent hierarchy, spacing, alignment, typography, and usability
    over decorative effects.
-   Avoid excessive gradients, glow, glass effects, and unnecessary
    animation. Use a restrained color palette with strong contrast.
-   Use skeleton animations as loading states. Skeletons must resemble
    the actual result layout, not just a generic spinning indicator.
-   Include responsive desktop, tablet, and mobile layouts.
-   Include accessible labels, keyboard interaction, visible focus
    states, semantic elements, and reduced-motion support.
-   Keep components reusable and code maintainable.

## 3. Suggested Repository Structure

``` text
license-plate-recognition/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── features/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── styles/
│   │   └── App.jsx
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── image_preprocessing.py
│   │   │   ├── plate_detector.py
│   │   │   ├── plate_recognizer.py
│   │   │   └── matlab_adapter.py
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── matlab/
│   ├── module1_enhancement/
│   ├── module2_segmentation/
│   ├── module3_transformations/
│   └── toolbox_gui/
├── samples/
├── docs/
│   └── architecture.md
├── .env.example
├── .gitignore
└── README.md
```

Adapt the structure to the existing repository rather than forcing it if
a sensible structure already exists.

## 4. Frontend Experience

Create a cohesive application with these areas:

### Main dashboard

-   A clear application header with a simple icon and project name.
-   A concise explanation of the workflow.
-   A prominent upload panel supporting drag-and-drop and browse.
-   Show accepted file formats and maximum upload size based on backend
    configuration.
-   A clear primary action to start detection.
-   Do not start processing before the user selects a valid image and
    requests it.

### Processing state

-   Show a skeleton version of the results panel while the API request
    is running.
-   Skeletons should have subtle shimmer/pulse animation.
-   Include a short status label such as "Analyzing image".
-   Avoid layout shifts when switching from skeleton to results.
-   Respect `prefers-reduced-motion`.

### Results

Show, where available: - Recognized license plate text with a copy
action. - OCR confidence only when the OCR engine returns a meaningful
score. - Original image. - Annotated image with the detected plate
bounding box. - Cropped plate preview. - Processing duration only when
measured by the backend. - Detection status and useful metadata.

Add a clear no-plate-detected state. Do not show fake plate text, fake
confidence, or sample metrics as if they were real results.

### Error states

Handle: - Unsupported file type. - File too large. - Invalid or
corrupted image. - API unavailable. - Processing failure. - No plate
detected.

Messages should be understandable and should tell the user what they can
do next.

### Visual system

-   Load DM Sans reliably, with sensible fallback fonts.
-   Use Lucide icons for upload, image, scan, settings, history, copy,
    download, alert, and navigation where appropriate.
-   Establish consistent spacing, type scale, border, radius, and shadow
    tokens.
-   Keep the layout clean and uncluttered.
-   Use a restrained palette; maintain readable contrast.
-   Avoid emoji, novelty fonts, and decorative icon clutter.

## 5. Backend Requirements

Use FastAPI with a clean separation of concerns.

Implement: - `GET /health` - `POST /api/v1/detect`

The detection endpoint should accept an image as `multipart/form-data`.

Validate: - File extension. - MIME type as one validation signal, not
the only one. - Decoded image content. - Configurable maximum file
size. - Empty uploads.

Use OpenCV to decode and process the image. Never trust the client
filename as a filesystem path. Prefer in-memory processing; if temporary
files are necessary, use generated names and clean them up.

Return a consistent response schema. Example only:

``` json
{
  "success": true,
  "message": "License plate processed",
  "result": {
    "plate_text": "ABC1234",
    "raw_ocr_text": "ABC 1234",
    "confidence": null,
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

Do not hardcode the example text, confidence, bounding box, or
processing time. Populate fields only from actual computation. If a
value is unavailable, return null or omit it according to the schema.

Use proper HTTP status codes and safe, user-friendly error responses. Do
not return internal stack traces.

## 6. Computer-Vision Pipeline

Implement the pipeline in small testable functions:

1.  Decode the uploaded image.
2.  Preserve original image dimensions and source image.
3.  Convert to grayscale when useful.
4.  Apply optional denoising and contrast enhancement.
5.  Locate candidate license-plate regions.
6.  Select or rank candidate regions using documented criteria.
7.  Crop the plate.
8.  Apply optional perspective/orientation correction if feasible.
9.  Resize and prepare the crop for OCR.
10. Run OCR.
11. Normalize the recognized string conservatively.
12. Return detection coordinates, text, and available metadata.
13. Create an annotated preview image.

Start with a simple, explainable OpenCV baseline detector. Keep the
detector interface modular so a YOLO model can be added later. Do not
download or claim to use a pretrained model unless it is actually
available and configured. Document the baseline's limitations.

OCR: - Prefer one OCR engine for the first working version. - Keep OCR
code behind a service interface. - Make language configuration
explicit. - Preserve raw OCR output for debugging and evaluation. - Do
not claim exact recognition if the output is uncertain.

## 7. MATLAB Course Modules

Create MATLAB source files and documentation for the three required
modules.

### Module 1: Image Enhancement

Implement and document: - Histogram equalization and histogram
matching. - Brightness and contrast using gray-level transformations. -
Mean and median smoothing. - Laplacian and high-pass sharpening. -
Fourier low-pass, high-pass, and band-pass filtering.

### Module 2: Segmentation and Edge Detection

Implement and document: - Sobel, Prewitt, and Canny edge detection. -
Otsu thresholding. - A basic region-based method such as region growing,
if feasible. - RGB/HSV color segmentation. - Dilation, erosion, opening,
and closing. - Shape-based segmentation where appropriate.

### Module 3: Geometric Transformations and Interpolation

Implement and document: - Translation, scaling, rotation, and affine
transformations. - Forward and reverse mapping. - Radial distortion
correction if calibration information is available; otherwise clearly
document assumptions and limitations. - Nearest-neighbor, bilinear, and
bicubic interpolation. - Image registration as optional work only.

Create a MATLAB GUI that lets a user load an image, select an operation,
and view the result, in line with the course brief.

Do not create empty placeholder functions and claim they are complete.
If a technique is not implemented, mark it as TODO in documentation.

## 8. MATLAB-Python Integration Boundary

Create a small `matlab_adapter.py` interface, but make it optional and
disabled by default.

If MATLAB Engine for Python is installed and configured, selected MATLAB
functions may be called through the adapter. If it is not available: -
The Python API must still start. - The main OpenCV detection flow must
still work. - The UI must not imply MATLAB processing occurred. - Return
a clear capability/status message where relevant.

Do not assume MATLAB Engine or MATLAB licensing is available in a cloud
container. Document separate-demo and runtime-integration deployment
options.

## 9. Testing

Add backend tests for: - Health endpoint. - Empty upload. - Unsupported
file. - Corrupted image. - Oversized file. - No-plate image. - Valid
image pipeline behavior using a test fixture. - Consistent response
schema.

Add frontend checks where practical: - Upload validation. - Loading
skeleton. - Successful result. - No-detection state. - API error
state. - Responsive layout.

Do not write tests that depend on a real external model download unless
the test explicitly handles the missing model.

## 10. Deployment and Developer Experience

Provide: - `.env.example` with documented variables. - `.gitignore` for
virtual environments, node modules, caches, generated images, and
secrets. - Backend and frontend setup instructions. - Production build
instructions. - CORS configuration through environment variables. - A
Dockerfile or compose setup only if it can be made accurate for the
selected dependencies. - A deployment guide explaining frontend/backend
separation and MATLAB constraints.

Never commit `.env`, secrets, private images, or large model weights
without explicit reason.

## 11. Documentation

Create or update: - `README.md`: overview, features, stack, setup, API,
MATLAB assignment alignment, testing, deployment, and limitations. -
`docs/architecture.md`: architecture diagrams, module responsibilities,
request flow, API schema, data flow, deployment, and integration
options. - Add comments to non-obvious computer-vision operations. -
Keep documentation synchronized with actual implementation.

## 12. Execution Rules

Work in this order:

1.  Inspect the current repository and report what already exists.
2.  Present a short implementation plan.
3.  Scaffold or adapt the backend.
4.  Implement and test health and upload validation.
5.  Implement the OpenCV pipeline.
6.  Integrate OCR.
7.  Build the frontend and its loading/error/result states.
8.  Add MATLAB modules and GUI.
9.  Add optional MATLAB adapter and explain its requirements.
10. Run tests and fix issues.
11. Update README and architecture documentation.
12. Prepare deployment configuration and exact deployment instructions.

Do not stop after generating a design mockup. Build working code.

Do not claim that a command, test, model, integration, or deployment
succeeded unless you actually ran it and verified the result. If an
external credential, MATLAB installation, OCR language package, model
file, or hosting account is missing, explain exactly what is needed and
continue with the parts that can be completed.

At the end, provide: - What was implemented. - Important files created
or changed. - Commands to run locally. - Tests executed and their actual
results. - Remaining limitations and manual steps. - Exact deployment
steps for the selected hosting setup.
