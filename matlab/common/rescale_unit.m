function out = rescale_unit(A)
%RESCALE_UNIT Linearly rescale an array to the range 0..1 (base MATLAB).
%   Replaces mat2gray. A constant array maps to all zeros.
A = double(A);
lo = min(A(:));
hi = max(A(:));
if hi > lo
    out = (A - lo) / (hi - lo);
else
    out = zeros(size(A));
end
end
