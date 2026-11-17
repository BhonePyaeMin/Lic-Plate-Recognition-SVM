function [bw, mag] = edge_detect(img, method, thresholdFraction)
%EDGE_DETECT Sobel, Prewitt or Canny edge detection.
%   [bw, mag] = edge_detect(img, 'sobel'|'prewitt'|'canny', thresholdFraction)
%
%   Sobel and Prewitt are implemented here with explicit 3x3 kernels: the
%   gradient magnitude sqrt(Gx^2 + Gy^2) is thresholded at
%   thresholdFraction * max(magnitude) (default 0.25). Canny uses the toolbox
%   edge() function (Gaussian smoothing, non-maximum suppression, hysteresis).
%   MAG is the normalised gradient magnitude (empty for Canny).
if nargin < 2, method = 'sobel'; end
if nargin < 3, thresholdFraction = 0.25; end
gray = im2double(to_gray(img));

switch lower(method)
    case 'sobel'
        gx = [-1 0 1; -2 0 2; -1 0 1];
    case 'prewitt'
        gx = [-1 0 1; -1 0 1; -1 0 1];
    case 'canny'
        if numel(thresholdFraction) == 2     % [low high] hysteresis thresholds, 0..1
            bw = edge(gray, 'canny', thresholdFraction);
        else
            bw = edge(gray, 'canny');
        end
        mag = [];
        return
    otherwise
        error('edge_detect:method', 'Unknown method "%s".', method);
end
gy = gx';
Gx = imfilter(gray, gx, 'replicate');
Gy = imfilter(gray, gy, 'replicate');
mag = sqrt(Gx .^ 2 + Gy .^ 2);
if max(mag(:)) > 0, mag = mag / max(mag(:)); end
bw = mag >= thresholdFraction;
end

function g = to_gray(img)
if size(img, 3) == 3, g = rgb2gray(img); else, g = img; end
end
