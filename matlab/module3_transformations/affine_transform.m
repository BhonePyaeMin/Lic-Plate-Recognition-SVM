function out = affine_transform(img, A, method, mapping, outSize)
%AFFINE_TRANSFORM Apply a 3x3 affine matrix to an image.
%   out = affine_transform(img, A, method, mapping, outSize)
%
%   A       3x3 matrix, [x_out; y_out; 1] = A * [x_in; y_in; 1], with x =
%           column and y = row (y grows downwards). Last row must be [0 0 1].
%   method  'nearest' | 'bilinear' | 'bicubic' (reverse mapping only)
%   mapping 'reverse' (default) or 'forward'
%   outSize [rows cols] of the output (default: same as the input)
%
%   Reverse mapping visits every OUTPUT pixel, computes where it comes from
%   with inv(A) and interpolates: no holes, supports all interpolators.
%   Forward mapping pushes every INPUT pixel to round(A * p): simple, but
%   it leaves holes when enlarging and overwrites pixels when shrinking.
%   It is provided to demonstrate that artefact.
if nargin < 3, method = 'bilinear'; end
if nargin < 4, mapping = 'reverse'; end
if nargin < 5, outSize = [size(img, 1) size(img, 2)]; end
if ~isequal(size(A), [3 3]) || ~isequal(A(3, :), [0 0 1])
    error('affine_transform:matrix', 'A must be a 3x3 affine matrix with last row [0 0 1].');
end
if abs(det(A)) < eps
    error('affine_transform:singular', 'A is singular and cannot be inverted.');
end

switch lower(mapping)
    case 'reverse'
        [U, V] = meshgrid(1:outSize(2), 1:outSize(1));
        src = A \ [U(:)'; V(:)'; ones(1, numel(U))];
        X = reshape(src(1, :), outSize(1), outSize(2));
        Y = reshape(src(2, :), outSize(1), outSize(2));
        out = sample_image(img, X, Y, method);
    case 'forward'
        [H, W, C] = size(img);
        [X, Y] = meshgrid(1:W, 1:H);
        dst = A * [X(:)'; Y(:)'; ones(1, numel(X))];
        u = round(dst(1, :));  v = round(dst(2, :));
        ok = u >= 1 & u <= outSize(2) & v >= 1 & v <= outSize(1);
        out = zeros([outSize C], class(img));
        srcIdx = find(ok);
        for c = 1:C
            plane = zeros(outSize, class(img));
            ch = img(:, :, c);
            plane(sub2ind(outSize, v(srcIdx), u(srcIdx))) = ch(srcIdx);
            out(:, :, c) = plane;
        end
    otherwise
        error('affine_transform:mapping', 'Unknown mapping "%s".', mapping);
end
end
