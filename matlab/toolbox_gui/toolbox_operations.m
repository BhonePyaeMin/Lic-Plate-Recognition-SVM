function ops = toolbox_operations()
%TOOLBOX_OPERATIONS Registry of operations shown in the MATLAB GUI.
%   Each entry has: name, module, params (struct array with label/default)
%   and run, a function handle  @(img, p, ref) -> result  where p is a numeric
%   vector of the (up to three) parameter values and ref is an optional
%   reference image (only histogram matching uses it).
%
%   To add a technique: implement it in its module folder, then append one
%   line here. The GUI needs no other change.
P = @(varargin) struct('label', varargin(1:2:end), 'default', varargin(2:2:end));
ops = struct('name', {}, 'module', {}, 'params', {}, 'run', {});

% ---- Module 1: enhancement ------------------------------------------------
ops(end+1) = op('Histogram equalization', 1, P(), @(i, p, r) histogram_equalization(i));
ops(end+1) = op('Histogram matching (needs reference)', 1, P(), @(i, p, r) require_ref(i, r));
ops(end+1) = op('Brightness / contrast', 1, P('Contrast', 1.3, 'Brightness', 0), @(i, p, r) adjust_brightness_contrast(i, p(1), p(2)));
ops(end+1) = op('Gamma correction', 1, P('Gamma', 0.7), @(i, p, r) gamma_correction(i, p(1)));
ops(end+1) = op('Mean smoothing', 1, P('Kernel size', 5), @(i, p, r) mean_smoothing(i, p(1)));
ops(end+1) = op('Median smoothing', 1, P('Kernel size', 3), @(i, p, r) median_smoothing(i, p(1)));
ops(end+1) = op('Laplacian sharpening', 1, P('Strength', 1), @(i, p, r) laplacian_sharpen(i, p(1)));
ops(end+1) = op('High-pass / high-boost sharpening', 1, P('Kernel size', 5, 'Boost', 1.5), @(i, p, r) highpass_sharpen(i, p(1), p(2)));
ops(end+1) = op('Fourier low-pass (Gaussian)', 1, P('Cut-off radius', 40), @(i, p, r) fourier_filter(i, 'low', p(1)));
ops(end+1) = op('Fourier high-pass (Gaussian)', 1, P('Cut-off radius', 30), @(i, p, r) fourier_filter(i, 'high', p(1)));
ops(end+1) = op('Fourier band-pass (Gaussian)', 1, P('Inner radius', 20, 'Outer radius', 80), @(i, p, r) fourier_filter(i, 'band', p(1), p(2)));

% ---- Module 2: segmentation and edges -------------------------------------
ops(end+1) = op('Sobel edges', 2, P('Threshold (0-1)', 0.25), @(i, p, r) edge_detect(i, 'sobel', p(1)));
ops(end+1) = op('Prewitt edges', 2, P('Threshold (0-1)', 0.25), @(i, p, r) edge_detect(i, 'prewitt', p(1)));
ops(end+1) = op('Canny edges', 2, P(), @(i, p, r) edge_detect(i, 'canny'));
ops(end+1) = op('Otsu threshold', 2, P(), @(i, p, r) otsu_threshold(i));
ops(end+1) = op('Region growing (centre seed)', 2, P('Tolerance', 20), @(i, p, r) region_growing(i, [], p(1)));
ops(end+1) = op('RGB segmentation (bright pixels)', 2, P('Min level', 170), @(i, p, r) color_segment_rgb(i, [p(1) p(1) p(1)], [255 255 255]));
ops(end+1) = op('HSV segmentation (low saturation)', 2, P('Max saturation', 0.2, 'Min value', 0.65), @(i, p, r) color_segment_hsv(i, [0 1], [0 p(1)], [p(2) 1]));
ops(end+1) = op('Dilation (Otsu mask)', 2, P('Size', 5), @(i, p, r) morphology_ops(i, 'dilate', p(1)));
ops(end+1) = op('Erosion (Otsu mask)', 2, P('Size', 5), @(i, p, r) morphology_ops(i, 'erode', p(1)));
ops(end+1) = op('Opening (Otsu mask)', 2, P('Size', 5), @(i, p, r) morphology_ops(i, 'open', p(1)));
ops(end+1) = op('Closing (Otsu mask)', 2, P('Size', 5), @(i, p, r) morphology_ops(i, 'close', p(1)));
ops(end+1) = op('Plate-shaped regions', 2, P('Min aspect', 2, 'Max aspect', 6), @(i, p, r) shape_segmentation(i, p(1), p(2)));

% ---- Module 3: geometry and interpolation ---------------------------------
ops(end+1) = op('Translate (reverse mapping)', 3, P('Shift x', 40, 'Shift y', 20), @(i, p, r) translate_image(i, p(1), p(2)));
ops(end+1) = op('Translate (forward mapping)', 3, P('Shift x', 40, 'Shift y', 20), @(i, p, r) translate_image(i, p(1), p(2), 'nearest', 'forward'));
ops(end+1) = op('Scale (nearest)', 3, P('Factor', 2), @(i, p, r) scale_image(i, p(1), p(1), 'nearest'));
ops(end+1) = op('Scale (bilinear)', 3, P('Factor', 2), @(i, p, r) scale_image(i, p(1), p(1), 'bilinear'));
ops(end+1) = op('Scale (bicubic)', 3, P('Factor', 2), @(i, p, r) scale_image(i, p(1), p(1), 'bicubic'));
ops(end+1) = op('Zoom in on a point (keeps frame size)', 3, P('Factor', 3, 'Centre x (%)', 50, 'Centre y (%)', 50), @(i, p, r) zoom_image(i, p(1), p(2), p(3), 'bicubic'));
ops(end+1) = op('Rotate (bilinear)', 3, P('Angle (deg, CCW)', 15), @(i, p, r) rotate_image(i, p(1), 'bilinear', 'reverse', true));
ops(end+1) = op('Rotate (forward mapping, shows holes)', 3, P('Angle (deg, CCW)', 15), @(i, p, r) rotate_image(i, p(1), 'nearest', 'forward'));
ops(end+1) = op('Shear (horizontal)', 3, P('Shear', 0.2), @(i, p, r) shear_image(i, p(1)));
ops(end+1) = op('Radial distortion correction (manual k1, k2)', 3, P('k1', 0.1, 'k2', 0), @(i, p, r) radial_distortion_correction(i, p(1), p(2)));
end

function s = op(name, module, params, fn)
s = struct('name', name, 'module', module, 'params', params, 'run', fn);
end

function out = require_ref(img, ref)
if isempty(ref)
    error('toolbox_operations:reference', 'Load a reference image first (Load reference button).');
end
out = histogram_matching(img, ref);
end
