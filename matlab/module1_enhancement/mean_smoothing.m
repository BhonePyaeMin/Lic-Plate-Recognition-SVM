function out = mean_smoothing(img, k)
%MEAN_SMOOTHING K-by-K box (mean) filter with replicated borders.
if nargin < 2, k = 5; end
k = max(1, round(k));
kernel = ones(k) / (k * k);
out = filter_replicate(to_uint8(img), kernel);
end
