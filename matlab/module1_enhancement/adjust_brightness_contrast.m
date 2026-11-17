function out = adjust_brightness_contrast(img, contrast, brightness)
%ADJUST_BRIGHTNESS_CONTRAST Linear gray-level transform about mid-gray.
%   out = contrast * (in - 128) + 128 + brightness, clipped to [0, 255].
%   contrast > 1 stretches, < 1 compresses. brightness shifts gray levels.
if nargin < 2, contrast = 1.2; end
if nargin < 3, brightness = 0; end
f = double(im2uint8(img));
out = uint8(min(max(contrast * (f - 128) + 128 + brightness, 0), 255));
end
