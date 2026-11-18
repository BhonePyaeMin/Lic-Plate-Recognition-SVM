function out = morph_binary(bw, operation, shape, sizePx)
%MORPH_BINARY Binary dilation, erosion, opening and closing (base MATLAB).
%   out = morph_binary(bw, 'dilate'|'erode'|'open'|'close', shape, sizePx)
%
%   Replaces strel/imdilate/imerode/imopen/imclose.
%   shape 'rectangle': structuring element sizePx wide by round(sizePx/2)
%                      high (suits horizontal text). Pass a two-element
%                      [height width] to set both sides explicitly.
%   shape 'disk'     : all pixels within Euclidean radius sizePx of the
%                      centre. strel('disk') uses a polygon approximation, so
%                      the two can differ by a pixel at the rim.
%   Pixels outside the image count as background for dilation and as
%   foreground for erosion (the imerode convention), so erosion does not eat
%   into the image border.
bw = logical(bw);
sizePx = max(1, round(sizePx));
switch lower(shape)
    case 'rectangle'
        if numel(sizePx) == 2          % explicit [height width]
            K = true(sizePx(1), sizePx(2));
        else
            K = true(max(1, round(sizePx / 2)), sizePx);
        end
    case 'disk'
        [x, y] = meshgrid(-sizePx:sizePx);
        K = (x .^ 2 + y .^ 2) <= sizePx ^ 2;
    otherwise
        error('morph_binary:shape', 'Unknown structuring element "%s".', shape);
end
origin = floor((size(K) + 1) / 2);    % strel convention for even sizes

switch lower(operation)
    case 'dilate', out = dilate(bw, K, origin);
    case 'erode',  out = erode(bw, K, origin);
    case 'open',   out = dilate(erode(bw, K, origin), K, origin);
    case 'close',  out = erode(dilate(bw, K, origin), K, origin);
    otherwise
        error('morph_binary:operation', 'Unknown operation "%s".', operation);
end
end

function out = dilate(bw, K, origin)
% out(p) = OR over SE offsets b of bw(p - (b - origin)); computed as a
% convolution, cropped so the SE origin lines up with the pixel.
[H, W] = size(bw);
full = conv2(double(bw), double(K));
out = full(origin(1):origin(1) + H - 1, origin(2):origin(2) + W - 1) > 0.5;
end

function out = erode(bw, K, origin)
% Erosion is the dual of dilation: erode(A, B) = not dilate(not A, reflect(B)).
% Padding with zeros inside dilate(not A) makes the outside count as foreground.
Kr = rot90(K, 2);
originR = size(K) - origin + 1;
out = ~dilate(~bw, Kr, originR);
end
