function out = median_smoothing(img, k)
%MEDIAN_SMOOTHING K-by-K median filter applied per channel.
%   Removes salt-and-pepper noise while preserving edges better than a mean
%   filter. Uses median_filter (base MATLAB, no toolbox); an even K is made
%   odd by adding 1.
if nargin < 2, k = 3; end
k = max(1, round(k));
img = to_uint8(img);
out = img;
for c = 1:size(img, 3)
    out(:, :, c) = median_filter(img(:, :, c), k);
end
end
