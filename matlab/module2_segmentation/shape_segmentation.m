function [mask, stats] = shape_segmentation(img, minAspect, maxAspect, minArea)
%SHAPE_SEGMENTATION Keep connected regions that look like a licence plate.
%   Pipeline: gray -> Sobel vertical edges -> Otsu -> horizontal closing to
%   fuse characters -> remove small blobs -> keep regions whose bounding box
%   aspect ratio is in [minAspect, maxAspect] and whose extent (area /
%   bounding-box area) is at least 0.4.
%   Defaults: aspect 2..6, minimum area 0.05 percent of the image.
%   Limitation: single-row plates only; cluttered backgrounds add false
%   regions. This mirrors the idea behind the Python baseline detector but
%   is a separate, simplified implementation.
if nargin < 2, minAspect = 2; end
if nargin < 3, maxAspect = 6; end
if size(img, 3) == 3, gray = rgb2gray(img); else, gray = img; end
gray = im2double(gray);
if nargin < 4, minArea = 0.0005 * numel(gray); end

% Plates are rich in vertical strokes: respond to horizontal gradients.
gx = filter_replicate(gray, [-1 0 1; -2 0 2; -1 0 1]);
edges = otsu_threshold(uint8(255 * rescale_unit(abs(gx))));
% Closing with a 3-row rectangle that is wide enough to join neighbouring characters.
fused = morph_binary(edges, 'close', 'rectangle', [3 max(9, round(size(gray, 2) / 40))]);
fused = remove_small_regions(fused, round(minArea), 8);

% Region measurements from the label image (replaces bwconncomp/regionprops).
[L, n] = label_components(fused, 8);
if n == 0
    mask = false(size(L));
    stats = struct('BoundingBox', {}, 'Area', {}, 'Extent', {});
    return
end
[r, c] = find(L > 0);
lab = L(L > 0);
area = accumarray(lab, 1, [n 1]);
minR = accumarray(lab, r, [n 1], @min);   maxR = accumarray(lab, r, [n 1], @max);
minC = accumarray(lab, c, [n 1], @min);   maxC = accumarray(lab, c, [n 1], @max);
width = maxC - minC + 1;
height = maxR - minR + 1;
extent = area ./ (width .* height);
aspect = width ./ height;
keep = aspect >= minAspect & aspect <= maxAspect & extent >= 0.4;

% Same fields as regionprops; BoundingBox is [x y width height] with the
% upper-left corner at the pixel edge (-0.5), as regionprops reports it.
ids = find(keep);
stats = struct('BoundingBox', cell(numel(ids), 1), 'Area', [], 'Extent', []);
for i = 1:numel(ids)
    j = ids(i);
    stats(i).BoundingBox = [minC(j) - 0.5, minR(j) - 0.5, width(j), height(j)];
    stats(i).Area = area(j);
    stats(i).Extent = extent(j);
end
mask = L > 0 & keep(max(L, 1));
end
