function run_operation_file(inPath, outPath, opId, paramsJson, refPath)
%RUN_OPERATION_FILE Entry point used by the Python backend (MATLAB Engine).
%   run_operation_file(inPath, outPath, opId, paramsJson, refPath)
%
%   Reads an image file, runs the operation named OPID with the parameters in
%   PARAMSJSON (a JSON object), and writes the result as a PNG to OUTPATH.
%   Operation ids and parameter keys are defined in
%   backend/app/services/filters.py and must stay in sync with this switch.
%   Binary results are written as 0/255 uint8. REFPATH is optional (histogram
%   matching). Image registration is intentionally not available here.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(root, 'common'), fullfile(root, 'module1_enhancement'), fullfile(root, 'module2_segmentation'), ...
        fullfile(root, 'module3_transformations'));

img = read_rgb_or_gray(inPath);
ref = [];
if nargin >= 5 && ~isempty(refPath)
    ref = read_rgb_or_gray(refPath);
end
p = struct();
if nargin >= 4 && ~isempty(paramsJson)
    p = jsondecode(paramsJson);
end

switch opId
    % ---- Module 1
    case 'hist_equalization',  out = histogram_equalization(img);
    case 'hist_matching'
        if isempty(ref), error('run_operation_file:ref', 'A reference image is required.'); end
        out = histogram_matching(img, ref);
    case 'brightness_contrast', out = adjust_brightness_contrast(img, p.contrast, p.brightness);
    case 'gamma',              out = gamma_correction(img, p.gamma);
    case 'mean',               out = mean_smoothing(img, p.kernel);
    case 'median',             out = median_smoothing(img, p.kernel);
    case 'laplacian',          out = laplacian_sharpen(img, p.strength);
    case 'highpass',           out = highpass_sharpen(img, p.kernel, p.boost);
    case 'fourier_low',        out = fourier_filter(img, 'low', p.cutoff, 2 * p.cutoff, p.shape);
    case 'fourier_high',       out = fourier_filter(img, 'high', p.cutoff, 2 * p.cutoff, p.shape);
    case 'fourier_band',       out = fourier_filter(img, 'band', p.inner, p.outer, p.shape);
    % ---- Module 2
    case 'sobel',              out = edge_detect(img, 'sobel', p.threshold);
    case 'prewitt',            out = edge_detect(img, 'prewitt', p.threshold);
    case 'canny',              out = edge_detect(img, 'canny', [p.low p.high]);
    case 'otsu',               out = otsu_threshold(img);
    case 'region_growing'
        [H, W, ~] = size(img);
        seed = [round(1 + p.seed_y / 100 * (H - 1)), round(1 + p.seed_x / 100 * (W - 1))];
        out = region_growing(img, seed, p.tolerance);
    case 'rgb_seg'
        out = color_segment_rgb(to_rgb(img), [p.r_min p.g_min p.b_min], [p.r_max p.g_max p.b_max]);
    case 'hsv_seg'
        out = color_segment_hsv(to_rgb(img), [p.h_min p.h_max] / 360, [p.s_min p.s_max], [p.v_min p.v_max]);
    case 'dilate',             out = morphology_ops(img, 'dilate', p.size, p.shape);
    case 'erode',              out = morphology_ops(img, 'erode', p.size, p.shape);
    case 'open',               out = morphology_ops(img, 'open', p.size, p.shape);
    case 'close',              out = morphology_ops(img, 'close', p.size, p.shape);
    case 'plate_shapes',       out = shape_segmentation(img, p.min_aspect, p.max_aspect);
    % ---- Module 3
    case 'translate',          out = translate_image(img, p.tx, p.ty, p.interpolation, p.mapping);
    case 'scale'
        if isfield(p, 'mode') && strcmp(p.mode, 'zoom-in')
            out = zoom_image(img, p.factor, p.center_x, p.center_y, p.interpolation);
        else
            out = scale_image(img, p.factor, p.factor, p.interpolation);
        end
    case 'rotate',             out = rotate_image(img, p.angle, p.interpolation, p.mapping, strcmp(p.enlarge, 'yes'));
    case 'shear',              out = shear_image(img, p.shear_x, p.shear_y, p.interpolation);
    case 'radial',             out = radial_distortion_correction(img, p.k1, p.k2, p.interpolation);
    otherwise
        error('run_operation_file:op', 'Unknown or unsupported operation "%s".', opId);
end

if islogical(out)
    out = uint8(out) * 255;
end
imwrite(out, outPath);
end

function img = read_rgb_or_gray(path)
img = imread(path);
if size(img, 3) == 4, img = img(:, :, 1:3); end
end

function img = to_rgb(img)
if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
end
