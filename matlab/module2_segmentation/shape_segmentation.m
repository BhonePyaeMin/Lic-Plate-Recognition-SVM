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
gx = imfilter(gray, [-1 0 1; -2 0 2; -1 0 1], 'replicate');
edges = otsu_threshold(uint8(255 * mat2gray(abs(gx))));
fused = imclose(edges, strel('rectangle', [3 max(9, round(size(gray, 2) / 40))]));
fused = bwareaopen(fused, round(minArea));

cc = bwconncomp(fused);
props = regionprops(cc, 'BoundingBox', 'Area', 'Extent');
keep = false(numel(props), 1);
for i = 1:numel(props)
    bb = props(i).BoundingBox;           % [x y width height]
    aspect = bb(3) / bb(4);
    keep(i) = aspect >= minAspect && aspect <= maxAspect && props(i).Extent >= 0.4;
end
stats = props(keep);
mask = ismember(labelmatrix(cc), find(keep));
end
