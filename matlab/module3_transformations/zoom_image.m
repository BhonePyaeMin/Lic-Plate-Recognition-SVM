function out = zoom_image(img, factor, centerXPct, centerYPct, method)
%ZOOM_IMAGE Magnify (factor > 1) or shrink (factor < 1) about a point, keeping the frame size.
%   out = zoom_image(img, factor, centerXPct, centerYPct, method)
%
%   The centre is given as a percentage of the width and height (50, 50 is the
%   middle of the picture). The pixel at the centre stays where it is and
%   everything else moves away from it (zoom in) or towards it (zoom out), so
%   zooming in on a licence plate brings the plate up close without making the
%   picture bigger. METHOD is 'nearest', 'bilinear' or 'bicubic'. Reverse
%   mapping is used: no holes. Positions outside the original picture (zooming
%   out) are black.
%
%   Affine matrix (x = column, y = row, 1-based pixel centres):
%       [x_out; y_out; 1] = [f 0 cx*(1-f); 0 f cy*(1-f); 0 0 1] * [x_in; y_in; 1]
if nargin < 3 || isempty(centerXPct), centerXPct = 50; end
if nargin < 4 || isempty(centerYPct), centerYPct = 50; end
if nargin < 5, method = 'bilinear'; end
if factor <= 0, error('zoom_image:factor', 'The zoom factor must be positive.'); end
[H, W, ~] = size(img);
cx = 1 + centerXPct / 100 * (W - 1);
cy = 1 + centerYPct / 100 * (H - 1);
A = [factor 0 cx * (1 - factor); 0 factor cy * (1 - factor); 0 0 1];
out = affine_transform(img, A, method, 'reverse', [H W]);
end
