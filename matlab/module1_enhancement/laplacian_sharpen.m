function out = laplacian_sharpen(img, strength)
%LAPLACIAN_SHARPEN Sharpen with the 4-neighbour Laplacian.
%   out = f - strength * laplacian(f). The kernel has a negative centre, so
%   subtracting the response adds edge detail back to the image.
if nargin < 2, strength = 1; end
kernel = [0 1 0; 1 -4 1; 0 1 0];
f = im2double(img);
out = f;
for c = 1:size(f, 3)
    lap = imfilter(f(:, :, c), kernel, 'replicate');
    out(:, :, c) = f(:, :, c) - strength * lap;
end
out = im2uint8(min(max(out, 0), 1));
end
