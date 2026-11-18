function run_all_tests()
%RUN_ALL_TESTS Self-checks for the three modules using synthetic images.
%   Prints PASS/FAIL per check and raises an error if any check fails.
%   Uses base MATLAB only (no toolbox). No sample files are required.
%
%   STATUS: all checks passed on MATLAB R2026a (base MATLAB, no toolbox).
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'common'), fullfile(root, 'module1_enhancement'), fullfile(root, 'module2_segmentation'), ...
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
    'median_filter matches a naive median (k=3,15)',  @t_median_exact
    'filter_replicate matches a naive correlation',   @t_filter_exact
    'morph dilate/erode match naive definitions',     @t_morph_exact
    'label_components 8- vs 4-connectivity',          @t_label
    'canny_edges hugs a rectangle outline',           @t_canny
    'enlarging leaves no black frame (3 methods)',    @t_no_frame
    'zoom_image keeps size and the centre pixel',     @t_zoom
    'run_operation_file scale zoom-in mode',          @t_dispatcher_zoom
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
ok = abs(mean_all(m) - mean_all(ref)) < abs(mean_all(g) - mean_all(ref));
end
function ok = t_brightness()
g = test_image();
ok = mean_all(adjust_brightness_contrast(g, 1, 30)) > mean_all(g);
end
function ok = t_median()
g = uint8(100 * ones(30));
g(10, 10) = 255;
m = median_smoothing(g, 3);
ok = m(10, 10) == 100;
end
function ok = t_mean()
g = test_image();
ok = std_all(mean_smoothing(g, 5)) < std_all(g);
end
function ok = t_laplacian()
rgb = cat(3, test_image(), test_image(), test_image());
s = laplacian_sharpen(rgb);
ok = isequal(size(s), size(rgb)) && isa(s, 'uint8');
end
function ok = t_fourier_low()
rng(2);
g = uint8(255 * rand(64));
ok = std_all(fourier_filter(g, 'low', 8)) < std_all(g);
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

function m = mean_all(x)
% Mean of all elements (base-MATLAB stand-in for mean2).
m = mean(double(x(:)));
end

function s = std_all(x)
% Standard deviation of all elements (base-MATLAB stand-in for std2).
s = std(double(x(:)));
end

% ----- tests for the base-MATLAB helpers in matlab/common ---------------------
function ok = t_median_exact()
rng(2);
A = uint8(randi(255, 17, 23));
ok = true;
for k = [3 15]
    r = (k - 1) / 2;
    ri = mirror_index(size(A, 1), r);  ci = mirror_index(size(A, 2), r);
    P = A(ri, ci);
    naive = zeros(size(A), 'uint8');
    for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            w = P(i:i + k - 1, j:j + k - 1);
            naive(i, j) = median(w(:));
        end
    end
    ok = ok && isequal(median_filter(A, k), naive);
end
end

function idx = mirror_index(n, r)
i = (1 - r):(n + r);
m = mod(i - 1, 2 * n);
idx = m + 1;  over = m >= n;  idx(over) = 2 * n - m(over);
end

function ok = t_filter_exact()
rng(3);
f = rand(12, 14);
K = [-1 0 1; -2 0 2; -1 0 1];
naive = zeros(size(f));
for i = 1:12
    for j = 1:14
        s = 0;
        for a = 1:3
            for b = 1:3
                s = s + K(a, b) * f(min(max(i + a - 2, 1), 12), min(max(j + b - 2, 1), 14));
            end
        end
        naive(i, j) = s;
    end
end
ok = max(abs(naive(:) - reshape(filter_replicate(f, K), [], 1))) < 1e-12;
end

function ok = t_morph_exact()
rng(4);
bw = rand(20, 25) > 0.7;
K = true(max(1, round(5 / 2)), 5);
o = floor((size(K) + 1) / 2);
[H, W] = size(bw);
dil = false(H, W);  ero = true(H, W);
for i = 1:H
    for j = 1:W
        for a = 1:size(K, 1)
            for b = 1:size(K, 2)
                ii = i - (a - o(1));  jj = j - (b - o(2));
                if ii >= 1 && ii <= H && jj >= 1 && jj <= W && bw(ii, jj), dil(i, j) = true; end
                ii = i + (a - o(1));  jj = j + (b - o(2));
                if ii >= 1 && ii <= H && jj >= 1 && jj <= W && ~bw(ii, jj), ero(i, j) = false; end
            end
        end
    end
end
ok = isequal(morph_binary(bw, 'dilate', 'rectangle', 5), dil) ...
  && isequal(morph_binary(bw, 'erode', 'rectangle', 5), ero);
end

function ok = t_label()
M = false(8, 10);
M(2:3, 2:3) = true;  M(4, 4) = true;  M(6, 6:9) = true;  M(1, 10) = true;
[~, n8] = label_components(M, 8);
[~, n4] = label_components(M, 4);
ok = n8 == 3 && n4 == 4;
end

function ok = t_canny()
G = zeros(80, 100);  G(25:55, 30:70) = 1;
E = canny_edges(G, [0.1 0.3]);
[r, c] = find(E);
ok = nnz(E) > 100 && all(r >= 22 & r <= 58 & c >= 27 & c <= 73) && ~any(E(30:50, 35:65), 'all');
end

% ----- scaling, zoom and borders ---------------------------------------------
function ok = t_no_frame()
img = uint8(200 * ones(30, 50, 3));
ok = true;
for m = {'nearest', 'bilinear', 'bicubic'}
    out = scale_image(img, 3.5, 3.5, m{1});
    ok = ok && all(out(:) == 200);   % any 0 would be a black edge band
end
end

function ok = t_zoom()
[X, Y] = meshgrid(1:80, 1:60);
img = uint8(mod(X + 2 * Y, 256));
out = zoom_image(img, 3, 50, 50, 'bilinear');
cx = 1 + 0.5 * 79;  cy = 1 + 0.5 * 59;     % the fixed point is between pixels, so compare its neighbourhood
ok = isequal(size(out), size(img)) && abs(double(out(30, 40)) - double(img(30, 40))) <= 3 ...
    && std(double(out(:))) > 0 && cx > 0 && cy > 0;
big = zoom_image(img, 4, 50, 50, 'nearest');
ok = ok && numel(unique(big(25:36, 35:46))) < numel(unique(img(25:36, 35:46)));   % 4x: fewer distinct values in a patch
end

function ok = t_dispatcher_zoom()
tmpIn = [tempname '.png'];  tmpOut = [tempname '.png'];
imwrite(test_image(), tmpIn);
cleanup = onCleanup(@() cellfun(@(f) delete(f), {tmpIn, tmpOut}));
params = '{"factor":3,"mode":"zoom-in","center_x":50,"center_y":50,"interpolation":"bicubic"}';
run_operation_file(tmpIn, tmpOut, 'scale', params, '');
a = imread(tmpIn);  b = imread(tmpOut);
ok = isequal(size(a, 1), size(b, 1)) && isequal(size(a, 2), size(b, 2));        % zoom-in keeps the frame size
params = '{"factor":3,"mode":"enlarge","center_x":50,"center_y":50,"interpolation":"bicubic"}';
run_operation_file(tmpIn, tmpOut, 'scale', params, '');
c = imread(tmpOut);
ok = ok && size(c, 1) == 3 * size(a, 1) && size(c, 2) == 3 * size(a, 2);       % enlarge mode grows the picture
end
