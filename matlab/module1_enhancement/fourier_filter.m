function out = fourier_filter(img, type, d1, d2, shape)
%FOURIER_FILTER Frequency-domain low-pass, high-pass or band-pass filter.
%   type  : 'low' | 'high' | 'band'
%   d1    : cut-off radius in pixels of the centred spectrum
%           (for 'band' it is the inner radius)
%   d2    : outer radius, used only by 'band'
%   shape : 'ideal' | 'gaussian'   (default 'gaussian'; ideal rings visibly)
%
%   The image is padded to avoid wrap-around, transformed, multiplied by a
%   radial transfer function H(D) and transformed back. 'high' and 'band'
%   remove the DC term, so those outputs are rescaled with rescale_unit for
%   display; 'low' is clipped to the original range.
if nargin < 2, type = 'low'; end
if nargin < 3, d1 = 30; end
if nargin < 4, d2 = 2 * d1; end
if nargin < 5, shape = 'gaussian'; end

img = to_uint8(img);
if size(img, 3) == 3
    out = img;
    for c = 1:3
        out(:, :, c) = fourier_filter(img(:, :, c), type, d1, d2, shape);
    end
    return
end

f = double(img);
[M, N] = size(f);
P = 2 * M;  Q = 2 * N;
F = fftshift(fft2(f, P, Q));
[u, v] = meshgrid(1:Q, 1:P);
D = sqrt((u - (Q / 2 + 1)) .^ 2 + (v - (P / 2 + 1)) .^ 2);

lowpass = @(d0) lowpass_response(D, d0, shape);
switch lower(type)
    case 'low'
        H = lowpass(d1);
    case 'high'
        H = 1 - lowpass(d1);
    case 'band'
        if d2 <= d1, error('fourier_filter:band', 'd2 must be larger than d1 for a band-pass filter.'); end
        H = lowpass(d2) - lowpass(d1);
    otherwise
        error('fourier_filter:type', 'Unknown filter type "%s".', type);
end

g = real(ifft2(ifftshift(F .* H)));
g = g(1:M, 1:N);
if strcmpi(type, 'low')
    out = uint8(min(max(g, 0), 255));
else
    out = to_uint8(rescale_unit(g));
end
end

function H = lowpass_response(D, d0, shape)
switch lower(shape)
    case 'ideal'
        H = double(D <= d0);
    case 'gaussian'
        H = exp(-(D .^ 2) / (2 * d0 ^ 2));
    otherwise
        error('fourier_filter:shape', 'Unknown filter shape "%s".', shape);
end
end
