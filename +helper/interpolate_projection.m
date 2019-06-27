function Pinterp = interpolate_projection(proj, metalTrace)
% projection linear interpolation
% Input:
% proj:         uncorrected projection
% metalTrace:   metal trace in projection domain (binary image)
% Output:
% Pinterp:      linear interpolation corrected projection

[NumofBin, NumofView] = size(proj);
Pinterp = zeros(NumofBin, NumofView);

for i = 1:NumofView
    mslice = metalTrace(:,i);
    pslice = proj(:,i);
    
    metalpos = find(mslice);
    nonmetalpos = find(mslice==0);
    
    pnonmetal = pslice(nonmetalpos);
    pslice(metalpos) = (interp1(nonmetalpos,pnonmetal,metalpos))';
    Pinterp(:,i) = pslice;
end
