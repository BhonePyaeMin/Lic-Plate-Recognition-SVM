function out = rotate_image(img, angleDeg, method, mapping, enlarge)
%ROTATE_IMAGE Rotate counter-clockwise by ANGLEDEG about the image centre.
%   enlarge = true grows the canvas so no corner is cut off (default false).
if nargin < 3, method = 'bilinear'; end
if nargin < 4, mapping = 'reverse'; end
if nargin < 5, enlarge = false; end
t = deg2rad(angleDeg);
H = size(img, 1);  W = size(img, 2);
% In image coordinates (y down) this matrix turns the picture counter-clockwise.
R = [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
if enlarge
    Hn = ceil(abs(H * cos(t)) + abs(W * sin(t)));
    Wn = ceil(abs(W * cos(t)) + abs(H * sin(t)));
else
    Hn = H;  Wn = W;
end
cIn  = [(W + 1) / 2; (H + 1) / 2];
cOut = [(Wn + 1) / 2; (Hn + 1) / 2];
Tin  = [1 0 -cIn(1); 0 1 -cIn(2); 0 0 1];
Tout = [1 0 cOut(1); 0 1 cOut(2); 0 0 1];
out = affine_transform(img, Tout * R * Tin, method, mapping, [Hn Wn]);
end
