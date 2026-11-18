function [L, n] = label_components(bw, conn)
%LABEL_COMPONENTS Label connected components of a binary image (base MATLAB).
%   [L, n] = label_components(bw, conn)
%
%   Replaces bwconncomp/labelmatrix. L has the size of BW with 0 for
%   background and 1..n for the n components. CONN is 4 or 8 (default 8).
%   Foreground pixels become graph nodes joined to their neighbours, and the
%   built-in conncomp (graph theory) finds the components without a
%   pixel-by-pixel MATLAB loop.
if nargin < 2, conn = 8; end
bw = logical(bw);
[H, W] = size(bw);
idx = find(bw);
n = 0;
L = zeros(H, W);
if isempty(idx), return, end

node = zeros(H * W, 1);
node(idx) = 1:numel(idx);
r = mod(idx - 1, H) + 1;
c = ceil(idx / H);

if conn == 4
    steps = [0 1; 1 0];                    % right, down
elseif conn == 8
    steps = [0 1; 1 0; 1 1; 1 -1];         % right, down, down-right, down-left
else
    error('label_components:conn', 'Connectivity must be 4 or 8.');
end

s = zeros(0, 1);
t = zeros(0, 1);
for k = 1:size(steps, 1)
    r2 = r + steps(k, 1);
    c2 = c + steps(k, 2);
    ok = r2 <= H & c2 >= 1 & c2 <= W;
    lin2 = zeros(size(idx));
    lin2(ok) = (c2(ok) - 1) * H + r2(ok);
    ok(ok) = bw(lin2(ok));                 % neighbour must be foreground too
    s = [s; node(idx(ok))];                %#ok<AGROW>
    t = [t; node(lin2(ok))];               %#ok<AGROW>
end

G = graph(s, t, [], numel(idx));
bins = conncomp(G);
L(idx) = bins;
n = max(bins);
end
