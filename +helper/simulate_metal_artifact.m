function [ma_sinogram_all, LI_sinogram_all, poly_sinogram, ma_CT_all, ...
        LI_CT_all, poly_CT, gt_CT, metal_trace_all] = simulate_metal_artifact( ...
        imgCT, imgMetalList, CTpara, MARpara)

% If we want Python hdf5 matrix to have size (N x H x W), 
% Matlab matrix should have size (W x H x N) 
% Therefore, we can permute (H, W, N) to (W x H x N)

n_mask = size(imgMetalList, 3);

%% tissue composition
MiuWater = MARpara.MiuWater;
threshWater = MARpara.threshWater;
threshBone = MARpara.threshBone;

img = imgCT / 1000 * MiuWater + MiuWater;
gt_CT = img';
imgWater = zeros(size(img));
imgBone = zeros(size(img));
bwWater = img <= threshWater;
bwBone = img >= threshBone;
bwBoth = im2bw(1 - bwWater - bwBone, 0.5);
imgWater(bwWater) = img(bwWater);
imgBone(bwBone) = img(bwBone);
imgBone(bwBoth) = (img(bwBoth) - threshWater) ./ (threshBone - threshWater) .* img(bwBoth);
imgWater(bwBoth) = img(bwBoth) - imgBone(bwBoth);

%% Synthesize non-metal poly CT
Pwater_kev = fanbeam(imgWater, CTpara.SOD,...
        'FanSensorGeometry', 'arc',...
        'FanSensorSpacing', CTpara.angSize, ...
        'FanRotationIncrement', 360/CTpara.angNum);
Pwater_kev = Pwater_kev * CTpara.imPixScale;
    
Pbone_kev = fanbeam(imgBone, CTpara.SOD,...
        'FanSensorGeometry', 'arc',...
        'FanSensorSpacing', CTpara.angSize, ...
        'FanRotationIncrement', 360/CTpara.angNum);
Pbone_kev = Pbone_kev * CTpara.imPixScale;

[NumofRou, NumofTheta] = size(Pwater_kev);
projkevAll = zeros(NumofRou, NumofTheta, 3);
projkevAll(:, :, 1) = Pwater_kev;
projkevAll(:, :, 2) = Pbone_kev;
projkvp = helper.pkev2kvp(projkevAll, MARpara.spectrum, MARpara.energies, MARpara.kev, MARpara.MiuAll);

% Poisson noise
scatterPhoton = 20;
temp = round(exp(-projkvp) .* MARpara.photonNum);
temp = temp + scatterPhoton;                               % simulate scattered photon
ProjPhoton = poissrnd(temp);
ProjPhoton(ProjPhoton == 0) = 1;
projkvpNoise = -log(ProjPhoton ./ MARpara.photonNum);

% correction
p1 = reshape(projkvpNoise, NumofRou.*NumofTheta, 1);
p1BHC = [p1  p1.^2  p1.^3] * MARpara.paraBHC;
poly_sinogram = reshape(p1BHC, NumofRou, NumofTheta);
% reconstruction
poly_CT = ifanbeam(poly_sinogram, CTpara.SOD,...
        'FanSensorGeometry', 'arc',...
        'FanSensorSpacing', CTpara.angSize,...
        'OutputSize', CTpara.imPixNum,...
        'FanRotationIncrement', 360 / CTpara.angNum);
poly_CT = poly_CT / CTpara.imPixScale;

poly_sinogram = single(poly_sinogram)';
poly_CT = single(poly_CT)';

%% Metal
ma_sinogram_all = single(zeros(CTpara.sinogram_size_x, CTpara.sinogram_size_y, n_mask));
LI_sinogram_all = single(zeros(CTpara.sinogram_size_x, CTpara.sinogram_size_y, n_mask));
metal_trace_all = single(zeros(CTpara.sinogram_size_x, CTpara.sinogram_size_y, n_mask));
ma_CT_all = single(zeros(CTpara.imPixNum, CTpara.imPixNum, n_mask));
LI_CT_all = single(zeros(CTpara.imPixNum, CTpara.imPixNum, n_mask));

parfor i = 1:n_mask
    imgMetal = squeeze(imgMetalList(:, :, i));
    imgMetal = imresize(imgMetal, [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear');
    Pmetal_kev = fanbeam(imgMetal, CTpara.SOD,...
            'FanSensorGeometry','arc',...
            'FanSensorSpacing', CTpara.angSize, ...
            'FanRotationIncrement',360/CTpara.angNum);
    metal_trace = Pmetal_kev > 0;
    Pmetal_kev = Pmetal_kev * CTpara.imPixScale;
    Pmetal_kev = MARpara.metalAtten * Pmetal_kev;

    % partial volume effect
    Pmetal_kev_bw =imerode(Pmetal_kev>0, [1 1 1]');
    Pmetal_edge = xor((Pmetal_kev>0), Pmetal_kev_bw);
    Pmetal_kev(Pmetal_edge) = Pmetal_kev(Pmetal_edge) / 4;

    % sinogram with metal
    projkevAllLocal = projkevAll;
    projkevAllLocal(:, :, 3) = Pmetal_kev;
    projkvpMetal = helper.pkev2kvp(projkevAllLocal, MARpara.spectrum, MARpara.energies, MARpara.kev, MARpara.MiuAll);
    temp = round(exp(-projkvpMetal) .* MARpara.photonNum);
    temp = temp + scatterPhoton;
    ProjPhoton = poissrnd(temp);
    ProjPhoton(ProjPhoton == 0) = 1;
    projkvpMetalNoise = -log(ProjPhoton ./ MARpara.photonNum);

    % correction
    p1 = reshape(projkvpMetalNoise, NumofRou.*NumofTheta, 1);
    p1BHC = [p1  p1.^2  p1.^3] * MARpara.paraBHC;
    ma_sinogram = reshape(p1BHC, NumofRou, NumofTheta);
    LI_sinogram = helper.interpolate_projection(ma_sinogram, metal_trace);

    % reconstruct   
    ma_CT = ifanbeam(ma_sinogram, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize,...
            'OutputSize', CTpara.imPixNum,...
            'FanRotationIncrement', 360 / CTpara.angNum);
    ma_CT = ma_CT / CTpara.imPixScale;
    
    LI_CT = ifanbeam(LI_sinogram, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize,...
            'OutputSize', CTpara.imPixNum,...
            'FanRotationIncrement', 360 / CTpara.angNum);
    LI_CT = LI_CT / CTpara.imPixScale;
    
    ma_sinogram_all(:, :, i) = ma_sinogram';
    LI_sinogram_all(:, :, i) = LI_sinogram';
    metal_trace_all(:, :, i) = metal_trace';
    ma_CT_all(:, :, i) = ma_CT';
    LI_CT_all(:, :, i) = LI_CT';

end


end