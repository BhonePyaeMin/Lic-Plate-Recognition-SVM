function bw = canny_edges(gray, thresholds)
%CANNY_EDGES Canny edge detector in base MATLAB.
%   bw = canny_edges(gray, [low high])
%
%   Replaces edge(gray, 'canny'). GRAY is a 2-D double image in 0..1.
%   Steps: (1) Gaussian-derivative gradients with sigma = sqrt(2) (the
%   edge() default), (2) gradient magnitude scaled so its maximum is 1,
%   (3) non-maximum suppression along the gradient direction quantised to
%   0, 45, 90 and 135 degrees, (4) hysteresis: pixels above HIGH seed the
%   result and any pixel above LOW that is 8-connected to a seed joins it.
%   Without thresholds, HIGH is the level below which 70 percent of pixels
%   fall and LOW = 0.4 * HIGH (the edge() rule). This is an independent
%   implementation, so output can differ slightly from edge().
if nargin < 2, thresholds = []; end
sigma = sqrt(2);
half = ceil(3 * sigma);
x = -half:half;
g = exp(-(x .^ 2) / (2 * sigma ^ 2));
g = g / sum(g);
dg = -x / sigma ^ 2 .* exp(-(x .^ 2) / (2 * sigma ^ 2));
dg = dg / sum(abs(dg) .* abs(x));          % unit response to a unit ramp

gx = filter_replicate(filter_replicate(gray, g(:)), dg);     % smooth rows, derive columns
gy = filter_replicate(filter_replicate(gray, g), dg(:));     % smooth columns, derive rows
mag = hypot(gx, gy);
mx = max(mag(:));
if mx > 0, mag = mag / mx; end

if numel(thresholds) == 2
    low = min(thresholds);  high = max(thresholds);
else
    counts = histcounts(mag(:), linspace(0, 1, 65));
    bin = find(cumsum(counts) > 0.7 * numel(mag), 1, 'first');
    if isempty(bin), bin = 64; end
    high = bin / 64;
    low = 0.4 * high;
end

% Non-maximum suppression: compare with the two neighbours on the gradient line.
angle = mod(atan2d(gy, gx), 180);
sector = zeros(size(mag));                          % 0 horizontal gradient
sector(angle >= 22.5 & angle < 67.5) = 1;           % 45 degrees (down-right)
sector(angle >= 67.5 & angle < 112.5) = 2;          % 90 degrees (vertical)
sector(angle >= 112.5 & angle < 157.5) = 3;         % 135 degrees (down-left)
M = zeros(size(mag) + 2);
M(2:end-1, 2:end-1) = mag;
c = M(2:end-1, 2:end-1);
n1 = zeros(size(mag));  n2 = zeros(size(mag));
shifts = {[0 -1], [0 1]; [-1 -1], [1 1]; [-1 0], [1 0]; [-1 1], [1 -1]};
for s = 0:3
    mask = sector == s;
    a = shifted(M, shifts{s + 1, 1});
    b = shifted(M, shifts{s + 1, 2});
    n1(mask) = a(mask);
    n2(mask) = b(mask);
end
thin = c >= n1 & c > n2;                           % asymmetric test thins plateaus

strong = thin & mag >= high;
weak = thin & mag >= low;
[L, n] = label_components(weak, 8);
if n == 0
    bw = false(size(mag));
    return
end
seeded = false(n, 1);
seeded(L(strong & L > 0)) = true;
bw = L > 0 & seeded(max(L, 1));
end

function S = shifted(M, d)
% Neighbour values of the unpadded image at offset d = [dRow dCol].
[Hp, Wp] = size(M);
S = M(2 + d(1):Hp - 1 + d(1), 2 + d(2):Wp - 1 + d(2));
end
