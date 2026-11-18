function out = morphology_ops(bw, operation, sizePx, shape)
%MORPHOLOGY_OPS Dilation, erosion, opening and closing of a binary image.
%   out = morphology_ops(bw, 'dilate'|'erode'|'open'|'close', sizePx, shape)
%   shape: 'rectangle' (default, sizePx wide by round(sizePx/2) high, which
%   suits horizontal text) or 'disk' (radius sizePx).
%   Non-binary input is binarised with Otsu's threshold first.
if nargin < 2, operation = 'close'; end
if nargin < 3, sizePx = 5; end
if nargin < 4, shape = 'rectangle'; end
if ~islogical(bw)
    bw = otsu_threshold(bw);
end
sizePx = max(1, round(sizePx));
if ~any(strcmpi(shape, {'rectangle', 'disk'}))
    error('morphology_ops:shape', 'Unknown structuring element "%s".', shape);
end
if ~any(strcmpi(operation, {'dilate', 'erode', 'open', 'close'}))
    error('morphology_ops:operation', 'Unknown operation "%s".', operation);
end
% common/morph_binary.m implements the four operations without the toolbox.
out = morph_binary(bw, operation, shape, sizePx);
end
