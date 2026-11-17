function run_all_tests()
%RUN_ALL_TESTS Self-checks for the three modules using synthetic images.
%   Prints PASS/FAIL per check and raises an error if any check fails.
%   Needs the Image Processing Toolbox. No sample files are required.
%
%   STATUS: written without access to MATLAB, so these checks have not been
%   executed yet. Run this file once in MATLAB and fix anything it reports.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'module1_enhancement'), fullfile(root, 'module2_segmentation'), ...
        fullfile(root, 'module3_transformations'), fullfile(root, 'toolbox_gui'), fullfile(root, 'api'));

checks = {
    'histogram_equalization widens gray range',       @t_hist_eq
    'histogram_matching moves mean toward reference', @t_hist_match
    'brightness raises the mean',                     @t_brightness
    'median filter removes a salt pixel',             @t_median
    'mean filter lowers variance',                    @t_mean
    'laplacian_sharpen keeps size and class',         @t_laplacian
    'fourier low-pass lowers variance',               @t_fourier_low
    'fourier band-pass rejects d2 <= d1',             @t_fourier_band_error
    'sobel finds rectangle edges',                    @t_sobel
    'otsu separates dark and bright regions',         @t_otsu
    'region growing fills a flat rectangle',          @t_region
    'hsv segmentation returns a logical mask',        @t_hsv
    'opening removes an isolated pixel',              @t_open
    'translate moves a pixel',                        @t_translate
    'scale by 2 doubles the size',                    @t_scale
    'rotate 90 equals rot90 on a square image',       @t_rotate90
    'identity affine is lossless (bilinear)',         @t_identity_bilinear
    'identity affine is lossless (bicubic)',          @t_identity_bicubic
    'forward mapping scaling leaves holes',           @t_forward_holes
    'radial correction with k1=k2=0 is identity',     @t_radial_identity
    'run_operation_file writes a result (mean)',      @t_dispatcher
    'run_operation_file rejects unknown ids',         @t_dispatcher_unknown
};
fails = 0;
for i = 1:size(checks, 1)
    try
        ok = checks{i, 2}();
    catch err
        ok = false;
        fprintf('       %s\n', err.message);
    end
    if ok
        fprintf('PASS   %s\n', checks{i, 1});
    else
        fprintf('FAIL   %s\n', checks{i, 1});
        fails = fails + 1;
    end
end
if fails > 0
    error('run_all_tests:failed', '%d check(s) failed.', fails);
end
fprintf('All %d checks passed.\n', size(checks, 1));
end

% ----- fixtures -------------------------------------------------------------
function g = test_image()
rng(1);
g = uint8(60 + 40 * rand(120, 200));
g(40:80, 60:140) = 200;
end

% ----- Module 1 -------------------------------------------------------------
function ok = t_hist_eq()
g = test_image();
e = histogram_equalization(g);
ok = double(max(e(:))) - double(min(e(:))) > double(max(g(:))) - double(min(g(:)));
end
function ok = t_hist_match()
g = test_image();
ref = uint8(150 + 50 * rand(120, 200));
m = histogram_matching(g, ref);
ok = abs(mean2(m) - mean2(ref)) < abs(mean2(g) - mean2(ref));
end
function ok = t_brightness()
g = test_image();
ok = mean2(adjust_brightness_contrast(g, 1, 30)) > mean2(g);
end
function ok = t_median()
g = uint8(100 * ones(30));
g(10, 10) = 255;
m = median_smoothing(g, 3);
ok = m(10, 10) == 100;
end
function ok = t_mean()
g = test_image();
ok = std2(mean_smoothing(g, 5)) < std2(g);
end
function ok = t_laplacian()
rgb = cat(3, test_image(), test_image(), test_image());
s = laplacian_sharpen(rgb);
ok = isequal(size(s), size(rgb)) && isa(s, 'uint8');
end
function ok = t_fourier_low()
rng(2);
g = uint8(255 * rand(64));
ok = std2(fourier_filter(g, 'low', 8)) < std2(g);
end
function ok = t_fourier_band_error()
try
    fourier_filter(test_image(), 'band', 30, 10);
    ok = false;
catch
    ok = true;
end
end

% ----- Module 2 -------------------------------------------------------------
function ok = t_sobel()
bw = edge_detect(test_image(), 'sobel', 0.5);
ok = nnz(bw) > 100;
end
function ok = t_otsu()
g = uint8(zeros(40));
g(:, 21:40) = 200;
[bw, level] = otsu_threshold(g);
ok = isequal(bw, g > 0) && level >= 0 && level < 200;
end
function ok = t_region()
g = uint8(50 * ones(60, 80));
g(20:40, 20:60) = 180;
mask = region_growing(g, [30 40], 10);
ok = isequal(mask, g == 180);
end
function ok = t_hsv()
rgb = cat(3, test_image(), test_image(), test_image());
ok = islogical(color_segment_hsv(rgb));
end
function ok = t_open()
bw = false(20);
bw(10, 10) = true;
ok = nnz(morphology_ops(bw, 'open', 3)) == 0;
end

% ----- Module 3 -------------------------------------------------------------
function ok = t_translate()
x = zeros(30, 30, 'uint8');
x(10, 10) = 255;                     % row 10, column 10
y = translate_image(x, 5, 3, 'nearest');
ok = y(13, 15) == 255 && nnz(y) == 1;
end
function ok = t_scale()
g = test_image();
ok = isequal(size(scale_image(g, 2, 2, 'bicubic')), 2 * size(g));
end
function ok = t_rotate90()
x = uint8(repmat(1:50, 50, 1) * 5);
r = rotate_image(x, 90, 'nearest');
ok = max(abs(double(r(:)) - double(reshape(rot90(x), [], 1)))) < 2;
end
function ok = t_identity_bilinear()
g = test_image();
ok = isequal(affine_transform(g, eye(3), 'bilinear'), g);
end
function ok = t_identity_bicubic()
g = test_image();
ok = isequal(affine_transform(g, eye(3), 'bicubic'), g);
end
function ok = t_forward_holes()
g = test_image();
out = affine_transform(g, [2 0 0; 0 2 0; 0 0 1], 'nearest', 'forward', [240 400]);
ok = nnz(out == 0) > 0.5 * numel(out);
end
function ok = t_dispatcher()
% Same entry point the Python backend calls through the MATLAB Engine.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'api'));
inPath = [tempname '.png'];  outPath = [tempname '.png'];
cleanup = onCleanup(@() cellfun(@(f) delete_if_exists(f), {inPath, outPath}));
imwrite(cat(3, test_image(), test_image(), test_image()), inPath);
run_operation_file(inPath, outPath, 'mean', '{"kernel":5}', '');
out = imread(outPath);
ok = isequal(size(out), [120 200 3]);
run_operation_file(inPath, outPath, 'sobel', '{"threshold":0.3}', '');
mask = imread(outPath);
ok = ok && all(ismember(unique(mask(:)), [0; 255]));
end
function ok = t_dispatcher_unknown()
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'api'));
inPath = [tempname '.png'];
imwrite(test_image(), inPath);
try
    run_operation_file(inPath, [tempname '.png'], 'nope', '{}', '');
    ok = false;
catch
    ok = true;
end
delete_if_exists(inPath);
end
function delete_if_exists(f)
if exist(f, 'file'), delete(f); end
end
function ok = t_radial_identity()
g = test_image();
ok = isequal(radial_distortion_correction(g, 0, 0), g);
end
