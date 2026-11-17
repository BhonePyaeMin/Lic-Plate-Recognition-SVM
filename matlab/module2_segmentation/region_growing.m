function mask = region_growing(img, seed, tolerance)
%REGION_GROWING Seeded region growing with 8-connectivity.
%   mask = region_growing(img, [row col], tolerance)
%   A neighbour joins the region when its gray level differs from the SEED's
%   gray level by at most TOLERANCE (0..255 scale), which avoids drift as
%   the region grows. The default seed is the image centre. Basic method: it
%   leaks through weak edges and is sensitive to the seed and tolerance.
if size(img, 3) == 3, img = rgb2gray(img); end
f = double(im2uint8(img));
[H, W] = size(f);
if nargin < 2 || isempty(seed), seed = [round(H / 2), round(W / 2)]; end
if nargin < 3, tolerance = 20; end
seed = min(max(round(seed), [1 1]), [H W]);

mask = false(H, W);
queue = zeros(H * W, 1);               % linear indices, preallocated FIFO
head = 1;  tail = 1;
s = sub2ind([H W], seed(1), seed(2));
queue(tail) = s;  tail = tail + 1;
mask(s) = true;
seedValue = f(s);
offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];

while head < tail
    cur = queue(head);  head = head + 1;
    [r, c] = ind2sub([H W], cur);
    for k = 1:size(offsets, 1)
        rr = r + offsets(k, 1);  cc = c + offsets(k, 2);
        if rr < 1 || rr > H || cc < 1 || cc > W, continue, end
        if mask(rr, cc), continue, end
        if abs(f(rr, cc) - seedValue) <= tolerance
            mask(rr, cc) = true;
            queue(tail) = sub2ind([H W], rr, cc);  tail = tail + 1;
        end
    end
end
end
