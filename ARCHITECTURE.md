# System Architecture

## Smart License Plate Detection and Recognition System

**Project type:** Web-based image-processing application\
**Primary backend:** Python FastAPI\
**Computer vision:** OpenCV\
**Recognition:** OCR engine (EasyOCR or Tesseract)\
**Course requirement:** MATLAB image-processing toolbox integration

------------------------------------------------------------------------

## 1. Project Overview

The system accepts a vehicle image, detects the license plate, processes
the plate region, recognizes its characters, and presents the result
through a modern web interface.

The project topic in the assignment is **License Plate Recognition
System Using OpenCV in Python**. The broader assignment also describes a
MATLAB-based toolbox and requires image enhancement, segmentation, edge
detection, morphological operations, geometric transformations,
interpolation, and an integrated MATLAB GUI.

To avoid silently ignoring either instruction, this architecture uses
Python as the web application and computer-vision service, while
retaining MATLAB processing functions as a documented and demonstrable
course module. Confirm with the lecturer whether MATLAB functions must
be called by the running web application or whether a separate MATLAB
toolbox demonstration is acceptable.

## 2. High-Level Architecture

``` mermaid
flowchart TD
    U[User] --> FE[Web Frontend]
    FE --> API[FastAPI Backend]
    API --> VAL[Upload Validation]
    VAL --> PIPE[Processing Pipeline]
    PIPE --> CV[OpenCV Detection and Preprocessing]
    CV --> OCR[OCR Recognition]
    PIPE -. Optional / course integration .-> MATLAB[MATLAB Toolbox Adapter]
    OCR --> POST[Result Validation and Formatting]
    MATLAB -. processed image / metrics .-> POST
    POST --> API
    API --> FE
    FE --> RES[Detection Result and Visualizations]
```

### Main layers

1.  **Presentation layer:** React or another modern frontend, with DM
    Sans typography, responsive layouts, icon-based navigation, and
    skeleton loading states.
2.  **API layer:** FastAPI receives uploads, validates requests,
    coordinates processing, and returns structured JSON.
3.  **Computer-vision layer:** OpenCV handles image loading,
    enhancement, plate localization, cropping, and image
    transformations.
4.  **Recognition layer:** OCR converts the cropped plate image into
    text.
5.  **MATLAB course module:** Implements the required image-processing
    techniques and exposes functions or a MATLAB GUI for demonstration.
    Integration method is subject to lecturer approval.
6.  **Result layer:** Returns the recognized text, confidence when
    available, bounding-box coordinates, and preview images.

## 3. Technology Stack

  -----------------------------------------------------------------------
  Area                    Recommended technology  Responsibility
  ----------------------- ----------------------- -----------------------
  Frontend                React + Vite            User interface

  Styling                 CSS / Tailwind CSS      Responsive visual
                                                  system

  Typography              DM Sans                 Consistent typography

  Icons                   Lucide icons            Navigation and controls

  Backend                 Python + FastAPI        REST API

  Image processing        OpenCV                  Detection and
                                                  preprocessing

  OCR                     EasyOCR or Tesseract    Character recognition

  MATLAB                  MATLAB Image Processing Required course
                          Toolbox                 techniques

  Data storage            SQLite (optional)       Store processing
                                                  history

  Deployment              Docker or separate      Repeatable deployment
                          frontend/backend        
                          hosting                 
  -----------------------------------------------------------------------

Choose one OCR engine for the first implementation. Keep the OCR
integration behind a small service interface so it can be replaced
later.

## 4. Request and Processing Flow

``` mermaid
sequenceDiagram
    participant User
    participant UI as Web UI
    participant API as FastAPI
    participant CV as OpenCV Pipeline
    participant OCR as OCR Engine

    User->>UI: Select and upload vehicle image
    UI->>UI: Validate file type and size
    UI->>API: POST /api/v1/detect
    API->>API: Validate and decode image
    API->>CV: Process image
    CV->>CV: Enhance and locate plate
    CV->>CV: Crop and prepare plate region
    CV->>OCR: Recognize plate characters
    OCR-->>CV: Text and confidence data
    CV-->>API: Detection result
    API-->>UI: JSON response and preview data
    UI-->>User: Show plate, image, and processing details
```

### Processing stages

1.  **Upload validation**
    -   Accept supported image formats such as JPG, JPEG, and PNG.
    -   Enforce a configurable upload-size limit.
    -   Reject invalid or unreadable files with a useful error message.
    -   Do not trust the client-provided filename or MIME type alone.
2.  **Image loading**
    -   Decode the image with OpenCV.
    -   Preserve the original image for comparison.
    -   Record image width and height.
3.  **Image enhancement**
    -   Convert to grayscale where appropriate.
    -   Apply optional denoising.
    -   Adjust contrast or use histogram equalization when useful.
    -   Keep the original image available for visual comparison.
4.  **License plate localization**
    -   Start with a clearly documented baseline detector, such as
        contour-based localization.
    -   Optionally support a trained detector such as YOLO as a later
        improvement.
    -   Return a bounding box for each detected plate.
    -   Do not claim a detection is correct when no plate is found.
5.  **Plate cropping and preparation**
    -   Crop the detected plate region.
    -   Correct orientation or perspective where feasible.
    -   Resize while preserving useful character detail.
    -   Apply thresholding and morphology as optional preprocessing
        steps.
6.  **OCR recognition**
    -   Pass the prepared crop to the selected OCR engine.
    -   Normalize whitespace and obvious formatting noise without
        inventing characters.
    -   Return the raw OCR output and normalized output separately where
        useful.
    -   Expose confidence only if the OCR engine provides it.
7.  **Result presentation**
    -   Draw the bounding box on a copy of the original image.
    -   Display the detected plate crop and recognized text.
    -   Show processing status, errors, and available confidence
        information.

## 5. MATLAB Toolbox Mapping

The assignment specifies three team modules. Keep these modules visible
in the codebase and report, even if the Python service performs the live
detection pipeline.

  -----------------------------------------------------------------------------------
  Assignment module       Required techniques                 Potential license-plate
                                                              use
  ----------------------- ----------------------------------- -----------------------
  Module 1: Image         Histogram equalization/matching,    Improve low-contrast,
  Enhancement             brightness and contrast,            noisy, or poorly
                          mean/median smoothing,              illuminated plate
                          Laplacian/high-pass sharpening,     images
                          Fourier low/high/band-pass          
                          filtering                           

  Module 2: Segmentation  Sobel, Prewitt, Canny, Otsu         Highlight plate
  and Edge Detection      thresholding, region-based and      boundaries and separate
                          RGB/HSV segmentation, dilation,     plate/character regions
                          erosion, opening, closing,          
                          shape-based segmentation            

  Module 3: Geometric     Translation, scaling, rotation,     Correct tilted or
  Transformations and     affine transformation,              distorted plate crops
  Interpolation           forward/reverse mapping, radial     and resize images
                          distortion correction,              
                          nearest-neighbor/bilinear/bicubic   
                          interpolation; optional             
                          registration                        
  -----------------------------------------------------------------------------------

### MATLAB integration options

**Option A --- Separate MATLAB demonstration (simplest):** - Build the
MATLAB functions and GUI required by the assignment. - Demonstrate the
functions on the same sample vehicle images. - Run the Python web
application separately for the end-to-end recognition demonstration. -
Clearly document that the two demonstrations are separate.

**Option B --- MATLAB called by the Python backend:** - Use MATLAB
Engine for Python if the environment and license support it. - Define a
small adapter that calls selected MATLAB functions. - Keep the FastAPI
route independent of MATLAB-specific code. - Provide a graceful error if
MATLAB Engine is unavailable. - Verify installation, licensing,
operating-system compatibility, and deployment support before choosing
this option.

Do not assume MATLAB Engine is available on a typical cloud host.
Confirm the deployment environment and lecturer expectations first.

## 6. API Design

Base path: `/api/v1`

  ------------------------------------------------------------------------
  Method                  Endpoint                 Purpose
  ----------------------- ------------------------ -----------------------
  GET                     `/health`                Check service
                                                   availability

  POST                    `/detect`                Upload an image and run
                                                   detection/OCR

  GET                     `/history`               List saved results,
                                                   only if history is
                                                   enabled

  GET                     `/history/{result_id}`   Retrieve one saved
                                                   result, only if history
                                                   is enabled
  ------------------------------------------------------------------------

### `POST /api/v1/detect`

**Request:** `multipart/form-data`

-   `file`: vehicle image
-   Optional processing parameters should be explicitly documented and
    validated.

**Example response:**

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

This is an illustrative response schema, not a claim that the system
already achieves this accuracy or speed. If confidence or processing
time is not available, return `null` or omit the field rather than
fabricate a value. If multiple plates are supported, return a list of
detections.

### Error response

``` json
{
  "success": false,
  "message": "No license plate detected",
  "error_code": "PLATE_NOT_FOUND"
}
```

Use suitable HTTP status codes, including 400 for invalid input, 413 for
oversized uploads, and 500 for unexpected server errors.

## 7. Suggested Repository Structure

``` text
license-plate-recognition/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── features/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── styles/
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
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

## 8. Frontend Design and Interaction Requirements

-   Use **DM Sans** throughout the application.
-   Use a restrained, professional palette with strong contrast and a
    clear visual hierarchy.
-   Use Lucide icons or another consistent icon set. Do not use emoji as
    interface icons.
-   Prefer clean cards, precise spacing, clear labels, and responsive
    layouts.
-   Make the upload area obvious and support drag-and-drop plus file
    selection.
-   Show the original image, detected image with bounding box, and
    cropped plate in a clear result layout.
-   Use skeleton animations while an image is being processed or results
    are loading. Skeletons should resemble the final content layout.
-   Respect reduced-motion preferences and avoid excessive animation.
-   Include empty, loading, success, no-detection, and error states.
-   Ensure controls have accessible labels, keyboard focus states, and
    sufficient contrast.

## 9. Security and Reliability

-   Validate file extension, MIME type, decoded image content, and file
    size.
-   Generate server-side temporary filenames; never use an uploaded
    filename as a filesystem path.
-   Store temporary files outside public static directories and remove
    them after processing.
-   Avoid logging uploaded image bytes or sensitive image contents.
-   Add request limits and safe error handling.
-   Keep secrets and environment-specific settings out of source
    control.
-   If history is enabled, explain what is stored and provide a
    retention/deletion approach.
-   Do not expose internal stack traces to users.

## 10. Testing and Evaluation

Test with: - Clear front-facing plates. - Tilted and rotated plates. -
Low-light and overexposed images. - Blurry and noisy images. - Different
plate sizes and backgrounds. - Images with no vehicle or no visible
plate. - Multiple vehicles, if multiple-plate detection is in scope.

Record: - Whether a plate was detected. - OCR exact-match rate on a
labeled test set. - Character-level accuracy, if practical. - False
detections and missed detections. - Processing time on the test
machine. - Results before and after selected preprocessing methods.

Do not report invented performance figures. Describe the dataset, test
conditions, and limitations.

## 11. Deployment Architecture

Recommended deployment split:

-   **Frontend:** deploy the built React application to a static
    frontend host.
-   **Backend:** deploy FastAPI to a Python-capable service or container
    host.
-   **Configuration:** set the backend API URL through a frontend
    environment variable.
-   **CORS:** allow only the actual frontend origin(s).
-   **Temporary storage:** use ephemeral storage unless persistent
    history is intentionally implemented.
-   **MATLAB:** deploy only if the selected host supports the required
    MATLAB runtime/Engine and licensing. Otherwise, keep MATLAB as a
    local course demonstration and document the boundary.

Before deployment: 1. Test the backend locally. 2. Test the frontend
against the local API. 3. Test upload size and invalid-file handling. 4.
Configure production environment variables and CORS. 5. Verify the
deployed `/health` endpoint. 6. Test a real image through the deployed
frontend. 7. Confirm that temporary uploads are not publicly accessible.

## 12. Implementation Milestones

1.  Create the repository and FastAPI health endpoint.
2.  Build the upload interface and API connection.
3.  Implement OpenCV image loading and preprocessing.
4.  Implement a baseline plate detector.
5.  Integrate OCR and return structured results.
6.  Build the result visualization and all UI states.
7.  Implement and test the three MATLAB course modules.
8.  Decide and verify the MATLAB integration option with the lecturer.
9.  Test on a documented sample set.
10. Deploy, write the report, and prepare the presentation.
