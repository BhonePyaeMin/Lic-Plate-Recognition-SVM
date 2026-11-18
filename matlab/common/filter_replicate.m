function out = filter_replicate(img, kernel)
%FILTER_REPLICATE Correlate an image with a kernel, replicating border pixels.
%   out = filter_replicate(img, kernel)
%
%   Base-MATLAB replacement for imfilter(img, kernel, 'replicate'). Works per
%   channel on 2-D or 3-D arrays. Integer input gives rounded, saturated
%   integer output of the same class; floating-point input stays floating.
%   Correlation (not convolution) is used, as imfilter does, so asymmetric
%   kernels such as Sobel keep their sign. For an even kernel the centre is
%   the element at floor((size + 1) / 2), the same convention as imfilter.
cls = class(img);
f = double(img);
[kh, kw] = size(kernel);
[H, W, C] = size(f);

padBeforeR = floor((kh + 1) / 2) - 1;   padAfterR = kh - floor((kh + 1) / 2);
padBeforeC = floor((kw + 1) / 2) - 1;   padAfterC = kw - floor((kw + 1) / 2);
rows = min(max((1 - padBeforeR):(H + padAfterR), 1), H);   % clamp = replicate
cols = min(max((1 - padBeforeC):(W + padAfterC), 1), W);

K = rot90(double(kernel), 2);   % conv2 convolves; flipping the kernel gives correlation
out = zeros(H, W, C);
for c = 1:C
    out(:, :, c) = conv2(f(rows, cols, c), K, 'valid');
end
if ~isfloat(img)
    out = cast(out, cls);       % cast rounds and saturates for integer classes
end
end
