function out = radial_distortion_correction(img, k1, k2, method)
%RADIAL_DISTORTION_CORRECTION Undo barrel/pincushion distortion (polynomial model).
%   out = radial_distortion_correction(img, k1, k2, method)
%
%   Model (normalised coordinates, origin at the image centre, radius scaled
%   so that the half-diagonal equals 1):
%       x_distorted = x_undistorted * (1 + k1*r^2 + k2*r^4),  r^2 = x^2 + y^2
%   For every pixel of the corrected image we compute where it lies in the
%   distorted input (reverse mapping) and interpolate.
%
%   ASSUMPTIONS AND LIMITATIONS
%   * K1 and K2 must come from a camera calibration (for example MATLAB
%     cameraCalibrator). No calibration data ships with this project, so the
%     GUI exposes them as manual values for demonstration only. Guessing
%     them does not recover the true geometry.
%   * Distortion centre = image centre (no principal-point offset).
%   * Tangential distortion is ignored.
%   Typical values: k1 < 0 corrects pincushion, k1 > 0 corrects barrel in
%   this convention.
if nargin < 2, k1 = 0.1; end
if nargin < 3, k2 = 0; end
if nargin < 4, method = 'bilinear'; end
[H, W, ~] = size(img);
cx = (W + 1) / 2;  cy = (H + 1) / 2;
norm = sqrt(cx ^ 2 + cy ^ 2);
[U, V] = meshgrid(1:W, 1:H);
xu = (U - cx) / norm;
yu = (V - cy) / norm;
r2 = xu .^ 2 + yu .^ 2;
factor = 1 + k1 * r2 + k2 * r2 .^ 2;
X = xu .* factor * norm + cx;
Y = yu .* factor * norm + cy;
out = sample_image(img, X, Y, method);
end
