clear; clc

%% Load params
config_file = fullfile('config', 'dataset.yaml');
splits = {'train', 'test'};

config = helper.YAML.read(config_file);
config = config.deep_lesion;

CTpara = config.CTpara;
names = fieldnames(CTpara);
for ii=1:numel(names)
    name = names{ii};
    p = CTpara.(name);
    if ischar(p)
        CTpara.(name) = eval(p);
    end
end

%% Load meta data
load(fullfile(config.mar_dir, 'SampleMasks'), 'CT_samples_bwMetal');
metal_masks = CT_samples_bwMetal;
MARpara = helper.get_mar_params(config.mar_dir);
data_list = importdata(config.data_list);

%% Generate MAR data
for phase=splits
    phase_dir = fullfile(config.dataset_dir, phase{1});

    image_indices = CTpara.([phase{1}, '_indices']);
    mask_indices = CTpara.([phase{1}, '_mask_indices']);

    image_size = [CTpara.imPixNum, CTpara.imPixNum, numel(mask_indices)];
    sinogram_size = [CTpara.sinogram_size_x, CTpara.sinogram_size_y, numel(mask_indices)];

    % prepare metal masks
    fprintf('Preparing metal masks...\n')
    selected_metal = metal_masks(:, :, mask_indices);
    mask_all = single(zeros(image_size));
    metal_trace_all = single(zeros(sinogram_size));

    for ii = 1:numel(mask_indices)
        mask_resize = imresize(selected_metal(:, :, ii), [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear');
        
        mask_proj = fanbeam(mask_resize, CTpara.SOD, ...
            'FanSensorGeometry', 'arc', ...
            'FanSensorSpacing', CTpara.angSize, ...
            'FanRotationIncrement', 360/CTpara.angNum);
        metal_trace = single(mask_proj > 0);
        
        mask_all(:, :, ii) = mask_resize';
        metal_trace_all(:, :, ii) = metal_trace';
    end

    for ii=1:numel(image_indices)
        image_name = data_list{image_indices(ii)};
        output_dir = fullfile(phase_dir, image_name(1:end-4));
        if ~isfolder(output_dir)
            mkdir(output_dir)
        end
        if isfile(fullfile(output_dir, 'gt.mat'))
            continue
        end

        fprintf('[%s][%d/%d] Processing %s\n', phase{1}, ii, numel(image_indices), image_name)
        raw_image = imread(fullfile(config.raw_dir, image_name));
 
        image = single(raw_image) - 32768;
        image = imresize(image, [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear');
        image(image < -1000) = -1000;

        [ma_sinogram_all, LI_sinogram_all, poly_sinogram, ma_CT_all, ...
        LI_CT_all, poly_CT, gt_CT, metal_trace_all] = helper.simulate_metal_artifact(...
            image, selected_metal, CTpara, MARpara);
        
        ct_file = fullfile(output_dir, 'gt.mat');
        image = imresize(raw_image, CTpara.imPixNum / size(raw_image, 1));
        save(ct_file, 'image');
        
        for jj=1:size(ma_CT_all, 3)
            ct_file = fullfile(output_dir, [num2str(jj), '.mat']);
            image = ma_CT_all(:, :, jj);
            save(ct_file, 'image');
        end
    end
end
