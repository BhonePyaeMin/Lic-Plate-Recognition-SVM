function out = gamma_correction(img, gamma)
%GAMMA_CORRECTION Power-law gray-level transform s = 255 * (r/255)^gamma.
%   gamma < 1 brightens shadows (useful for under-exposed plates),
%   gamma > 1 darkens highlights.
if nargin < 2, gamma = 0.7; end
out = uint8(255 * (double(to_uint8(img)) / 255) .^ gamma);
end
