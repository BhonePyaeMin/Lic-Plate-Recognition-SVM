function out = histogram_matching(img, ref)
%HISTOGRAM_MATCHING Match the histogram of IMG to that of REF (specification).
%   For each source level s, find the smallest reference level r whose
%   cumulative probability is >= that of s. RGB images are matched per
%   channel; a grayscale reference is replicated for an RGB source.
img = to_uint8(img);
ref = to_uint8(ref);
if size(img, 3) == 3
    if size(ref, 3) ~= 3
        ref = repmat(ref, [1 1 3]);
    end
    out = img;
    for c = 1:3
        out(:, :, c) = histogram_matching(img(:, :, c), ref(:, :, c));
    end
    return
end
if size(ref, 3) == 3
    ref = rgb2gray(ref);
end
cs = cumsum(accumarray(double(img(:)) + 1, 1, [256 1])) / numel(img);
cr = cumsum(accumarray(double(ref(:)) + 1, 1, [256 1])) / numel(ref);
lut = zeros(256, 1, 'uint8');
for s = 1:256
    r = find(cr >= cs(s) - eps, 1, 'first');
    if isempty(r), r = 256; end
    lut(s) = r - 1;
end
out = lut(double(img) + 1);
end
