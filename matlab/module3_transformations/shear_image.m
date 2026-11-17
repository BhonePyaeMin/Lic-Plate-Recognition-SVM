function out = shear_image(img, shx, shy, method)
%SHEAR_IMAGE Horizontal (SHX) and vertical (SHY) shear: a general affine example.
%   Horizontal shear shifts each row by shx * (y - centre row); useful to
%   undo the slant of a plate seen from the side.
if nargin < 3, shy = 0; end
if nargin < 4, method = 'bilinear'; end
H = size(img, 1);  W = size(img, 2);
cx = (W + 1) / 2;  cy = (H + 1) / 2;
S = [1 shx 0; shy 1 0; 0 0 1];
T1 = [1 0 -cx; 0 1 -cy; 0 0 1];
T2 = [1 0 cx; 0 1 cy; 0 0 1];
out = affine_transform(img, T2 * S * T1, method, 'reverse');
end
