# MATLAB image-processing toolbox

Course modules implemented as plain MATLAB functions plus a GUI.

**Status:** the module functions and `tests/run_all_tests.m` (30 checks) were run on MATLAB R2026a
and pass. The standalone GUI (`toolbox_gui/plate_toolbox_gui.m`) has **not** been run yet. No toolbox
is required for the modules: `common/` holds base-MATLAB replacements for the Image Processing
Toolbox functions that were used before (`im2uint8`, `mat2gray`, `imfilter`, `medfilt2`, `strel`,
`imdilate`/`imerode`/`imopen`/`imclose`, `edge` for Canny, `bwareaopen`, `bwconncomp`,
`regionprops`). They are independent implementations, so outputs can differ slightly from the
toolbox versions (for example the disk structuring element and Canny).

## Run

```matlab
cd matlab
addpath(genpath(pwd))
run_all_tests            % in tests/: self-checks on synthetic images
plate_toolbox_gui        % opens the GUI
```

## Contents

| Module | File | Technique |
|---|---|---|
| 1 | `histogram_equalization.m`, `histogram_matching.m` | Histogram equalization and matching (manual CDF) |
| 1 | `adjust_brightness_contrast.m`, `gamma_correction.m` | Gray-level transformations |
| 1 | `mean_smoothing.m`, `median_smoothing.m` | Spatial smoothing |
| 1 | `laplacian_sharpen.m`, `highpass_sharpen.m` | Laplacian and high-pass / high-boost sharpening |
| 1 | `fourier_filter.m` | Fourier low-pass, high-pass, band-pass (ideal or Gaussian) |
| 2 | `edge_detect.m` | Sobel and Prewitt (explicit kernels), Canny (toolbox `edge`) |
| 2 | `otsu_threshold.m` | Otsu thresholding from the histogram |
| 2 | `region_growing.m` | Seeded region growing, 8-connectivity |
| 2 | `color_segment_rgb.m`, `color_segment_hsv.m` | RGB box and HSV range segmentation |
| 2 | `morphology_ops.m` | Dilation, erosion, opening, closing |
| 2 | `shape_segmentation.m` | Plate-shaped region filtering (aspect ratio, extent) |
| 3 | `affine_transform.m`, `translate_image.m`, `scale_image.m`, `rotate_image.m`, `shear_image.m` | Affine transforms with forward and reverse mapping |
| 3 | `sample_image.m` | Nearest-neighbour, bilinear, bicubic interpolation (manual) |
| 3 | `radial_distortion_correction.m` | Polynomial radial model; needs calibration values |
| GUI | `toolbox_gui/plate_toolbox_gui.m`, `toolbox_operations.m` | Load image, choose operation, view result |

## Used by the web app

`api/run_operation_file.m` is the entry point the Python backend calls through MATLAB Engine. Its
`switch` must stay in sync with operation ids and parameter keys in
`backend/app/services/filters.py`. To make the web app use MATLAB: install MATLAB, run
`pip install matlabengine` in the backend environment, restart the backend (see the main README).

## Known gaps

* TODO: image registration (optional in the brief) is not implemented.
* Radial distortion correction needs coefficients from a real camera calibration; none are
  provided, so the GUI uses manual values for demonstration only.
* Canny, `imfilter`, `medfilt2`, `imdilate`/`imerode` and `regionprops` come from the toolbox
  rather than being re-implemented.

## Relationship to the web application

The web app (Python, OpenCV, FastAPI) does not call this code by default. See
`../docs/architecture.md` for the optional MATLAB Engine adapter and the two integration options.

## Module 3 notes

* `zoom_image(img, factor, centreX%, centreY%, method)` magnifies around a point and keeps the frame size (the web app's Scaling step has the same *zoom-in* mode).
* `sample_image` replicates the edge pixels into the outer half pixel. Earlier versions zeroed that band, which left a black frame around every enlarged image and made the MATLAB engine about 3 dB worse than OpenCV on a 4x enlargement; both engines now agree.
