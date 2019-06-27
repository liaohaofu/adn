function MARpara = get_mar_params(param_root)

% materials
load(fullfile(param_root, 'MiuofH2O.mat'), 'MiuofH2O')
load(fullfile(param_root, 'MiuofTi.mat'), 'MiuofTi')
load(fullfile(param_root, 'MiuofFe.mat'), 'MiuofFe')
load(fullfile(param_root, 'MiuofCu.mat'), 'MiuofCu')
load(fullfile(param_root, 'MiuofAu.mat'), 'MiuofAu')
load(fullfile(param_root, 'MiuofBONE_Cortical_ICRU44.mat'), 'MiuofBONE_Cortical_ICRU44')
% spectrum data
load(fullfile(param_root, 'GE14Spectrum120KVP.mat'), 'GE14Spectrum120KVP')


kVp = 120;
energies = 20:kVp;
kev = 70; 
photonNum = 2*10^(7);
materialID = 1;

threshWaterHU = 100;
threshBoneHU = 1500;
MiuWater = 0.192;
threshWater = threshWaterHU/1000*MiuWater + MiuWater;
threshBone = threshBoneHU/1000*MiuWater + MiuWater;

MiuofMetal = [];
MiuofMetal(:, :, 1) = MiuofTi(1:kVp, :);
MiuofMetal(:, :, 2) = MiuofFe(1:kVp, :);
MiuofMetal(:, :, 3) = MiuofCu(1:kVp, :);
MiuofMetal(:, :, 4) = MiuofAu(1:kVp, :);

densityMetal = [4.5 7.8 8.9 2];
metalAtten = densityMetal(materialID) * MiuofMetal(kev, 7, materialID);

% materials
MiuAll = [];
MiuAll(:, :, 1) = MiuofH2O(1:kVp, :);
MiuAll(:, :, 2) = MiuofBONE_Cortical_ICRU44(1:kVp, :);
MiuAll(:, :, 3) = MiuofMetal(1:kVp, :, materialID);

spectrum = GE14Spectrum120KVP(1:kVp, 2);

% water BHC
thickness = [0: 0.05: 50]';        % thickness of water, cm
pwaterkev = MiuofH2O(kev, 7)*thickness;
pwaterkvp = helper.pkev2kvp(pwaterkev, spectrum, energies, kev, MiuofH2O(1:kVp, :));
A = [pwaterkvp  pwaterkvp.^2  pwaterkvp.^3];
paraBHC = pinv(A)*pwaterkev;


% return
MARpara.kev = kev;
MARpara.spectrum = spectrum;
MARpara.energies = energies;
MARpara.photonNum = photonNum;
MARpara.MiuWater = MiuWater;
MARpara.MiuAll = MiuAll;
MARpara.threshWater = threshWater;
MARpara.threshBone = threshBone;
MARpara.paraBHC = paraBHC;
MARpara.metalAtten = metalAtten;

end