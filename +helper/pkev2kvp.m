% kev projection to kvp projection
%
% 2017-07-09

function projkvp = pkev2kvp(projkevAll, spectrum, energies, kev, MiuAll)

% projkevAll: all single materials' projection at given kev, 3d: (bin, view,
% material)
% spectrum
% energies: energies girds, e.g. [20:120]
% kev: kev under which projkev is obtained
% Miu: mass attenuation of the all given material over energies, 3d: (energy, mode, materials)

AttenuMode = 7;
matNum = size(projkevAll, 3);       % number of materials
projAll = zeros(size(projkevAll));
ProjEnergy = zeros(size(projkevAll, 1), size(projkevAll, 2));
projkvp = zeros(size(projkevAll, 1), size(projkevAll, 2));

for ien = energies
    for imat = 1:matNum
        % projection at current energy for each material component
        projAll(:, :, imat) = MiuAll(ien, AttenuMode, imat)/MiuAll(kev, AttenuMode, imat)*projkevAll(:, :, imat);
    end
    proj = sum(projAll, 3);
    Ptmp = spectrum(ien)*exp(-proj);    % according to the spectrum ratio
    ProjEnergy = ProjEnergy + Ptmp;      
end
ProjEnergyBlankRatio = sum(spectrum(energies))*ones(size(projkvp));
projkvp = -log(ProjEnergy./ProjEnergyBlankRatio);
