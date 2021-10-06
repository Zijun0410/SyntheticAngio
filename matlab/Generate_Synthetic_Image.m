%%% Synthetic the angiogram image based on the Rhino vessel image and the
%%% real background image. 

% Setting the path etc
% Generate the infor_saver_cell which contains
%     angio_struct with the following fields
%         angio_struct.meta_data = struct();
%         angio_struct.stenosis_data = struct();
%         angio_struct.endpoint_data = struct();
%         angio_struct.segment = struct();
%         angio_struct.volumn = struct();
%         angio_struct.branch_identifiers = branch_identifiers;
%         angio_struct.ref_size = ref_size;
Load_Rhino_Image

for iVess=1:length(infor_saver_cell)
    %%
    %-% Load information into angio_struct
    angio_struct = infor_saver_cell{iVess};
    real_image_fullpath = fullfile(real_image_path, angio_struct.file_name, 'frame1.png');
    back_image_path = fullfile(base_data_path, 'BackGround_Image', 'Clean', ...
        strcat(angio_struct.file_name,'.png'));
    angio_struct.background = im2double(imread(back_image_path));
    angio_struct.real_image = im2double(imread(real_image_fullpath));
    %-% Generate Synthetic angiogram images
    gaussian_factor = 1;
    rescale_factor = 0.3;
    border_threshold = 0.5;
    adjust_factor_1 = 0.02;
    random_factor = 100;
    background = angio_struct.background;
    for identifier = angio_struct.branch_identifiers
        volumn_image = angio_struct.volumn.(identifier{1});
        % blur_image = imgaussfilt(volumn_image, gaussian_factor);
        vessel_shade = (1 - volumn_image)*rescale_factor;
        % Get the background region where vessel_shade would lie
        vessel_shade_mask = volumn_image<0.99;
        background_shade = zeros(512,512)+1;
        background_shade(vessel_shade_mask) = background(vessel_shade_mask);
        if debug;figure;imshow(background_shade);end
        % The darker the backgound region (smaller the value), the less
        % vessel shade we need to add (smaller the value because we minus 
        % the shade at the next step). 
        % vessel_shade need to multiple a factor that represent the background
        % darkness, for those pixels that is small(darker) than the mean value of
        % the vessel backgound region. 
        % background_shade(background_shade > mean(background(vessel_shade_mask))) = 1;
        bakcground_corrected_vessel_shade = vessel_shade.*background_shade;
        blurred_shade = imgaussfilt(bakcground_corrected_vessel_shade, gaussian_factor);
        background = background - blurred_shade - rand(1)/random_factor;
    end
    image_display = cat(3, angio_struct.background, background, angio_struct.real_image);
    % visualize image in a horizental concatenated way
    figure; montage(image_display,'size',[1 3]);
end

%% 
% figure;imshow(hah);
% target_number = [0.1, 0.5, 1, 2];
% effect_display = zeros(angio_struct.ref_size, angio_struct.ref_size, length(target_number));
% for iFactor = 1:length(target_number)
%     volumn_image = angio_struct.volumn.major;
%     blur_image = imgaussfilt(volumn_image, target_number(iFactor), 'FilterSize',7);
%     effect_display(:,:,iFactor) = blur_image;
% end
% figure; montage(effect_display, 'Size', [1, length(target_number)]);

