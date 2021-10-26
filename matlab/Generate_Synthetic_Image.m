%%% Synthetic the angiogram image based on the Rhino vessel image and the
%%% real background image. 

% Setting the path etc
% Generate the infor_saver_cell which contains
%     angio_struct with the following fields
%         angio_struct.meta_data = struct();
%         angio_struct.stenosis_summary = table();
%         angio_struct.endpoint_data = struct();
%         angio_struct.segment = struct();
%         angio_struct.volumn = struct();
%         angio_struct.branch_identifiers = branch_identifiers;
%         angio_struct.ref_size = ref_size;
% Obtain stenosis_summary_tab

debug_flag = 0;
demo_flag = 0;
% batch_id = 'UKR';
batch_id = 'UoMR';
Load_Rhino_Image

%%
for iVess=1:length(infor_saver_cell)
    %%
    %-% Load information into angio_struct
    angio_struct = infor_saver_cell{iVess};
    if strcmp(batch_id, 'UKR')
        back_image_path = fullfile(base_data_path, 'BackGround_Image', 'Clean', ...
            strcat(angio_struct.file_name,'.png'));
    else
        meta_infor = meta_infors(contains(meta_infors.name_combine,angio_struct.file_name),:);
        % back_image_path = fullfile(meta_infor.save_dir{1}, meta_infor.background{1});
        save_dir = 'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\BackGround_Image\UoM_Right\';
        back_image_path = fullfile(save_dir, meta_infor.filename{1}, meta_infor.background{1});
    end
    angio_struct.background = im2double(imread(back_image_path));

    %-% Generate Synthetic angiogram images
    gaussian_factor = 1;
    rescale_factor = 0.3;
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
        if debug_flag;figure;imshow(background_shade);end
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
    angio_struct.synthetic_image = background;
    % Create Folder
    if ~isfolder(angio_struct.output_folder); mkdir(angio_struct.output_folder);end
    if demo_flag
        real_image_fullpath = fullfile(real_image_path, angio_struct.file_name, 'frame1.png');
        angio_struct.real_image = im2double(imread(real_image_fullpath));
        image_display = cat(3, angio_struct.background, angio_struct.synthetic_image, angio_struct.real_image);
        % Visualize the background image, sythetic and real image
        figure('visible','off');  fig = montage(image_display,'size',[1 3]);
        montage_image_data = fig.CData;
        % Write the montage image 
        imwrite(montage_image_data,fullfile(angio_struct.output_folder, 'montage.png'));
    end
    % Write the angio_struct to folder
    save(fullfile(angio_struct.output_folder, 'angio_struct.mat'),'angio_struct');
    % Write relevant information to file
    % 1. the synthetic image
    imwrite(angio_struct.synthetic_image, fullfile(angio_struct.output_folder, 'synthetic.png'));
    % 2. the stentosis information
    writetable(angio_struct.stenosis_summary,fullfile(angio_struct.output_folder,'stenosis_infor.csv'),'Delimiter',',')  
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

