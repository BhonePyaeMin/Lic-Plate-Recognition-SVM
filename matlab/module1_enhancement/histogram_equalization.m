function out = histogram_equalization(img)
%HISTOGRAM_EQUALIZATION Global histogram equalization (manual CDF mapping).
%   out = histogram_equalization(img) equalizes a grayscale image. For RGB
%   input only the V channel of HSV is equalized so colours are preserved.
%
%   Method: s = round(255 * (CDF(r) - CDFmin) / (N - CDFmin)).
if size(img, 3) == 3
    hsv = rgb2hsv(img);
    v = histogram_equalization(im2uint8(hsv(:, :, 3)));
    hsv(:, :, 3) = double(v) / 255;
    out = im2uint8(hsv2rgb(hsv));
    return
end
img = im2uint8(img);
counts = accumarray(double(img(:)) + 1, 1, [256 1]);
cdf = cumsum(counts);
cdfMin = cdf(find(cdf > 0, 1, 'first'));
n = numel(img);
if n == cdfMin          % constant image: nothing to equalize
    out = img;
    return
end
lut = uint8(round((cdf - cdfMin) / (n - cdfMin) * 255));
out = lut(double(img) + 1);
end
