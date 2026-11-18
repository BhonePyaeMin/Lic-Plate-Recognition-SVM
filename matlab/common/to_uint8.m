function out = to_uint8(img)
%TO_UINT8 Convert an image to uint8 using base MATLAB only (no toolbox).
%   Replaces im2uint8. uint8 is returned unchanged, logical maps to 0/255,
%   uint16 is scaled by 255/65535, and double/single are taken as 0..1 and
%   scaled to 0..255 with rounding and clipping.
switch class(img)
    case 'uint8'
        out = img;
    case 'logical'
        out = uint8(img) * 255;
    case 'uint16'
        out = uint8(round(double(img) / 257));
    case {'double', 'single'}
        out = uint8(round(min(max(double(img), 0), 1) * 255));
    otherwise
        error('to_uint8:class', 'Unsupported image class "%s".', class(img));
end
end
