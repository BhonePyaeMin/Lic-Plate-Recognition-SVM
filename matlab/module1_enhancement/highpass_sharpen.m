function out = highpass_sharpen(img, k, boost)
%HIGHPASS_SHARPEN High-boost sharpening in the spatial domain.
%   highpass = f - mean_k(f);  out = f + boost * highpass.
%   boost = 1 is unsharp masking; boost > 1 is high-boost filtering.
if nargin < 2, k = 5; end
if nargin < 3, boost = 1.5; end
f = im2double(img);
kernel = ones(k) / (k * k);
out = f;
for c = 1:size(f, 3)
    low = filter_replicate(f(:, :, c), kernel);
    out(:, :, c) = f(:, :, c) + boost * (f(:, :, c) - low);
end
out = to_uint8(min(max(out, 0), 1));
end
