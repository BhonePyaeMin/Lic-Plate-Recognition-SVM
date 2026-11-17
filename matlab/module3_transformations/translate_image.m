function out = translate_image(img, tx, ty, method, mapping)
%TRANSLATE_IMAGE Shift the image by TX pixels right and TY pixels down.
if nargin < 4, method = 'bilinear'; end
if nargin < 5, mapping = 'reverse'; end
A = [1 0 tx; 0 1 ty; 0 0 1];
out = affine_transform(img, A, method, mapping);
end
