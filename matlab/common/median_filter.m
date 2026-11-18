function out = median_filter(img, k)
%MEDIAN_FILTER K-by-K median filter of a 2-D uint8 image (base MATLAB).
%   out = median_filter(img, k)
%
%   Replaces medfilt2(img, [k k], 'symmetric'). K is forced odd (k + 1 when
%   even) so the window has a centre, which also matches the Python backend.
%   Borders use symmetric padding.
%
%   Two strategies, both exact:
%   * k <= 11: stack the k*k shifted copies and sort them (row blocks keep
%     memory bounded).
%   * k  > 11: sweep the 256 gray levels; for each level count how many
%     window pixels are <= level with a box sum from an integral image and
%     take the first level where the count reaches the median rank. The cost
%     does not grow with k.
img = to_uint8(img);
k = max(1, round(k));
if mod(k, 2) == 0, k = k + 1; end
if k == 1, out = img; return; end
r = (k - 1) / 2;
[H, W] = size(img);
P = img(symmetric_index(H, r), symmetric_index(W, r));   % (H+2r) by (W+2r)
rank = (k * k + 1) / 2;

if k <= 11
    out = zeros(H, W, 'uint8');
    block = max(1, floor(4e6 / (W * k * k)));
    for r0 = 1:block:H
        r1 = min(H, r0 + block - 1);
        stack = zeros(r1 - r0 + 1, W, k * k, 'uint8');
        idx = 0;
        for dy = 0:k - 1
            for dx = 0:k - 1
                idx = idx + 1;
                stack(:, :, idx) = P(r0 + dy:r1 + dy, 1 + dx:W + dx);
            end
        end
        sorted = sort(stack, 3);
        out(r0:r1, :) = sorted(:, :, rank);
    end
else
    out = zeros(H, W, 'uint8');
    done = false(H, W);
    for v = 0:255
        counts = box_sum(double(P <= v), k, H, W);
        hit = ~done & counts >= rank;
        out(hit) = v;
        done = done | hit;
        if all(done(:)), break; end
    end
end
end

function idx = symmetric_index(n, r)
% Index vector 1-r..n+r that mirrors across the border (edge pixel repeated).
i = (1 - r):(n + r);
m = mod(i - 1, 2 * n);
idx = m + 1;
over = m >= n;
idx(over) = 2 * n - m(over);
end

function s = box_sum(M, k, H, W)
% Sum over every k-by-k window of the padded matrix M (result is H by W).
S = zeros(size(M, 1) + 1, size(M, 2) + 1);
S(2:end, 2:end) = cumsum(cumsum(M, 1), 2);
s = S(k + 1:k + H, k + 1:k + W) - S(1:H, k + 1:k + W) ...
  - S(k + 1:k + H, 1:W) + S(1:H, 1:W);
end
