function mask = color_segment_hsv(img, hueRange, satRange, valRange)
%COLOR_SEGMENT_HSV Segmentation by hue, saturation and value ranges (0..1).
%   mask = color_segment_hsv(img, [hLo hHi], [sLo sHi], [vLo vHi])
%   If hLo > hHi the hue range wraps around 0 (for example red: [0.95 0.05]).
%   Defaults select low-saturation, bright pixels (white/grey plates).
if size(img, 3) ~= 3, error('color_segment_hsv:input', 'An RGB image is required.'); end
if nargin < 2, hueRange = [0 1]; end
if nargin < 3, satRange = [0 0.2]; end
if nargin < 4, valRange = [0.65 1]; end
hsv = rgb2hsv(im2double(img));
h = hsv(:, :, 1);
if hueRange(1) <= hueRange(2)
    hueOk = h >= hueRange(1) & h <= hueRange(2);
else
    hueOk = h >= hueRange(1) | h <= hueRange(2);
end
mask = hueOk & hsv(:, :, 2) >= satRange(1) & hsv(:, :, 2) <= satRange(2) ...
     & hsv(:, :, 3) >= valRange(1) & hsv(:, :, 3) <= valRange(2);
end
