function out = scale_image(img, sx, sy, method, mapping)
%SCALE_IMAGE Resize by factors SX (width) and SY (height); output size changes.
%   Uses the pixel-centre convention x_out = s * (x_in - 0.5) + 0.5 so the
%   image does not drift by half a pixel. Note: shrinking by a large factor
%   with nearest/bilinear interpolation aliases; low-pass filter first (see
%   mean_smoothing) if that matters.
if nargin < 3 || isempty(sy), sy = sx; end
if nargin < 4, method = 'bilinear'; end
if nargin < 5, mapping = 'reverse'; end
A = [sx 0 0.5 - 0.5 * sx; 0 sy 0.5 - 0.5 * sy; 0 0 1];
outSize = max(1, round([size(img, 1) * sy, size(img, 2) * sx]));
out = affine_transform(img, A, method, mapping, outSize);
end
