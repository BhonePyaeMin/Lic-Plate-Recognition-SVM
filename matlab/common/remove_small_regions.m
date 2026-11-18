function out = remove_small_regions(bw, minArea, conn)
%REMOVE_SMALL_REGIONS Delete connected regions smaller than MINAREA pixels.
%   out = remove_small_regions(bw, minArea, conn)
%   Base-MATLAB replacement for bwareaopen. CONN is 4 or 8 (default 8).
if nargin < 3, conn = 8; end
[L, n] = label_components(bw, conn);
if n == 0
    out = false(size(bw));
    return
end
area = accumarray(L(L > 0), 1, [n 1]);
keep = area >= minArea;
out = L > 0 & keep(max(L, 1));
end
