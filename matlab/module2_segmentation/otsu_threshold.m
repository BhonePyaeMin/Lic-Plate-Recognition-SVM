function [bw, level] = otsu_threshold(img)
%OTSU_THRESHOLD Otsu's global threshold, implemented from the histogram.
%   Chooses the level t (0..255) that maximises the between-class variance
%   sigma_b^2(t) = (mu_T*omega(t) - mu(t))^2 / (omega(t) * (1 - omega(t))).
%   Returns the binary image (pixels above t) and the level.
if size(img, 3) == 3, img = rgb2gray(img); end
img = to_uint8(img);
p = accumarray(double(img(:)) + 1, 1, [256 1]) / numel(img);
omega = cumsum(p);
mu = cumsum(p .* (0:255)');
muT = mu(end);
sigmaB2 = (muT * omega - mu) .^ 2 ./ (omega .* (1 - omega));
sigmaB2(~isfinite(sigmaB2)) = 0;
[~, idx] = max(sigmaB2);
level = idx - 1;
bw = img > level;
end
