function mask = color_segment_rgb(img, lower, upper)
%COLOR_SEGMENT_RGB Box segmentation in RGB space.
%   mask = color_segment_rgb(img, [rLo gLo bLo], [rHi gHi bHi]) with values
%   on the 0..255 scale. Defaults select bright, near-white pixels such as
%   a white plate background. RGB boxes are sensitive to illumination; see
%   color_segment_hsv for a more robust alternative.
if size(img, 3) ~= 3, error('color_segment_rgb:input', 'An RGB image is required.'); end
if nargin < 2, lower = [170 170 170]; end
if nargin < 3, upper = [255 255 255]; end
img = im2uint8(img);
mask = true(size(img, 1), size(img, 2));
for c = 1:3
    mask = mask & img(:, :, c) >= lower(c) & img(:, :, c) <= upper(c);
end
end
