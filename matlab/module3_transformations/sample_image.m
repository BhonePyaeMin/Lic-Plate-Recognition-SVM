function out = sample_image(img, X, Y, method)
%SAMPLE_IMAGE Sample IMG at fractional source coordinates (X = column, Y = row).
%   out = sample_image(img, X, Y, 'nearest'|'bilinear'|'bicubic')
%   X and Y are matrices of the output size. Pixel centres are at integer
%   coordinates; (1,1) is the centre of the top-left pixel, so the image
%   covers [0.5, W + 0.5] by [0.5, H + 0.5].
%
%   Border rule: a position inside the image area, including the outer half
%   pixel (as happens at the edge when enlarging), takes the value of the
%   nearest real pixel (replication). Only positions farther outside give 0
%   (black). Earlier versions zeroed that outer half pixel and extrapolated
%   there, which left a black frame around every enlarged image.
%
%   nearest  : value of the closest pixel (blocky, fastest).
%   bilinear : weighted average of the 2x2 neighbourhood.
%   bicubic  : cubic convolution (a = -0.5, Keys) over the 4x4 neighbourhood;
%              border neighbours are clamped. Can overshoot slightly, so the
%              result is clipped to the input data range.
if nargin < 4, method = 'bilinear'; end
class0 = class(img);
img = double(img);
[H, W, C] = size(img);
tol = 1e-6;   % absorb floating-point error on exact border coordinates
inside = X >= 0.5 - tol & X <= W + 0.5 + tol & Y >= 0.5 - tol & Y <= H + 0.5 + tol;
X = min(max(X, 1), W);   % replicate the edge pixels into the outer half pixel
Y = min(max(Y, 1), H);
out = zeros([size(X), C]);

for c = 1:C
    ch = img(:, :, c);
    switch lower(method)
        case 'nearest'
            xi = min(max(round(X), 1), W);
            yi = min(max(round(Y), 1), H);
            val = ch(sub2ind([H W], yi, xi));
        case 'bilinear'
            x0 = min(max(floor(X), 1), W - 1);  y0 = min(max(floor(Y), 1), H - 1);
            if W == 1 || H == 1, error('sample_image:size', 'Image must be at least 2x2.'); end
            dx = X - x0;  dy = Y - y0;
            v00 = ch(sub2ind([H W], y0,     x0));
            v01 = ch(sub2ind([H W], y0,     x0 + 1));
            v10 = ch(sub2ind([H W], y0 + 1, x0));
            v11 = ch(sub2ind([H W], y0 + 1, x0 + 1));
            val = (1 - dy) .* ((1 - dx) .* v00 + dx .* v01) ...
                +      dy  .* ((1 - dx) .* v10 + dx .* v11);
        case 'bicubic'
            x0 = floor(X);  y0 = floor(Y);
            dx = X - x0;    dy = Y - y0;
            val = zeros(size(X));
            for m = -1:2
                wy = cubic_weight(m - dy);
                yi = min(max(y0 + m, 1), H);
                for n = -1:2
                    wx = cubic_weight(n - dx);
                    xi = min(max(x0 + n, 1), W);
                    val = val + wy .* wx .* ch(sub2ind([H W], yi, xi));
                end
            end
            val = min(max(val, min(ch(:))), max(ch(:)));
        otherwise
            error('sample_image:method', 'Unknown interpolation method "%s".', method);
    end
    val(~inside) = 0;
    out(:, :, c) = val;
end
out = cast(out, class0);
end

function w = cubic_weight(s)
a = -0.5;
s = abs(s);
w = zeros(size(s));
near = s <= 1;
far = s > 1 & s < 2;
w(near) = (a + 2) * s(near) .^ 3 - (a + 3) * s(near) .^ 2 + 1;
w(far) = a * s(far) .^ 3 - 5 * a * s(far) .^ 2 + 8 * a * s(far) - 4 * a;
end
