function out = median_smoothing(img, k)
%MEDIAN_SMOOTHING K-by-K median filter applied per channel.
%   Removes salt-and-pepper noise while preserving edges better than a mean
%   filter.
if nargin < 2, k = 3; end
k = max(1, round(k));
img = im2uint8(img);
out = img;
for c = 1:size(img, 3)
    out(:, :, c) = medfilt2(img(:, :, c), [k k], 'symmetric');
end
end
