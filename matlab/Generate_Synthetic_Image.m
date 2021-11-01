%%% Synthetic the angiogram image based on the Rhino vessel image and the
%%% real background image. 
%%% Author: Zijun Gao
%%% Last Update: Oct 30th 2021
%%% Project: SyntheticAngio

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

% Configure the loading dir & stuff
Config_Path

%%%%%%%%%%% MAKE CHANGE IF NEEDED %%%%%%%%%%%%%%
batch_id = 'UoMR_Movement';
% batch_id = 'UoMR';
ref_size = 512;
branch_identifiers = {'major', 'branch_1', 'branch_2', 'branch_3', 'branch_4', 'branch_5'};
% Flag
debug_flag = 0;
demo_flag = 0;
save_struct_flag = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-% Some default dir setting
image_load_dir = fullfile(base_data_path, 'Rhino_Output', batch_id);
output_save_dir = fullfile(base_data_path, 'Sythetic_Output', batch_id);
if ~isfolder(output_save_dir)
    mkdir(output_save_dir)
end

% Define meta files
if contains(batch_id,'UoMR')
    meta_file_name = 'UoM_Right_endpoint.csv';
elseif contains(batch_id,'UKR')
    meta_file_name = 'meta_summary.csv';
end

%-% Load stenosis infor generated from Rhino-python scripts
stenosis_data = readtable(fullfile(image_load_dir, 'stnosis_infor.csv'));
stenosis_infors = sortrows(stenosis_data,1);
stenosis_infors.Properties.VariableNames = {'index', 'fileName', 'stenosis_count', ...
    'stenosis_location', 'effect_region', 'percentage', 'distanceSourceToDetector',... 
    'distanceSourceToPatient', 'positionerPrimaryAngle', 'positionerSecondaryAngle'};
unique_image_cases = unique(stenosis_infors.fileName);

%-% Load metedata generated from dicom file and manusal annotation
meta_infors = readtable(fullfile(base_data_path, 'Meta_Data', meta_file_name));

%% Iterate through cases
%-% Initiate stenosis summary table
stenosis_summary_tab = [];
%%
for iCase = 1:size(unique_image_cases,1)
    %-% Initate AngioStruct
    angio_struct = struct();
    angio_struct.meta_data = struct();
    angio_struct.endpoint_data = struct();
    angio_struct.segment = struct();
    angio_struct.volumn = struct();
    angio_struct.branch_identifiers = branch_identifiers;
    angio_struct.ref_size = ref_size;
    
    %-% Get case specifice stenosis_infor
    stenosis_infor = stenosis_infors(ismember(stenosis_infors.fileName,unique_image_cases{iCase}),:);
    if strcmp(batch_id,'UKR')
        file_name = stenosis_infor.fileName{1};
    else
    % file_name_dot = stenosis_infor.fileName{1};
    % file_name = file_name_dot(1:end-1);
        file_name = stenosis_infor.fileName{1};
    end
    
    %-% Record stenosis information 
    angio_struct.file_name = file_name;
    angio_struct.output_folder = fullfile(output_save_dir, file_name);
    stenosis_detail_col_names = {'stenosis_location', 'effect_region', 'percentage', 'degree', 'identifier'};
    stenosis_detail = zeros(size(stenosis_infor,1),4);
    for iStenosis = 1:size(stenosis_infor,1)
        stenosis_detail(iStenosis,1) = stenosis_infor.stenosis_location(iStenosis);
        stenosis_detail(iStenosis,2) = stenosis_infor.effect_region(iStenosis);
        stenosis_detail(iStenosis,3) = stenosis_infor.percentage(iStenosis);
        if stenosis_infor.percentage(iStenosis) < 0.5
            stenosis_detail(iStenosis,4) = 0;
        elseif stenosis_infor.percentage(iStenosis) < 0.7
            stenosis_detail(iStenosis,4) = 1;
        else
            stenosis_detail(iStenosis,4) = 2;
        end        
        stenosis_detail(iStenosis,5) = stenosis_infor.stenosis_count(iStenosis);
    end
    stenosis_detail = sortrows(array2table(stenosis_detail, 'VariableNames', ...
        stenosis_detail_col_names),5);
    path_combo = string(split(angio_struct.output_folder, '\'));
    stenosis_detail.output_folder = join(path_combo(2:end), '+');
    
    %-% Display information when running
    disp(['Running the ', num2str(iCase), ' case. File name: ', file_name])
    file_png_folder = fullfile(image_load_dir, file_name);
    
    %-% Get the corresponding meta data and the center of catheter endpoint 
    if contains(batch_id,'UKR')
        meta_infor = meta_infors(contains(meta_infors.FileName,file_name),:);
        x_center = meta_infor.CenterX;
        y_center = meta_infor.CenterY; 
    else
        meta_infor = meta_infors(ismember(meta_infors.name_combine,file_name),:);
        if length(meta_infor.x) >1
            disp('More than one match, check here!')
        end
        x_center = meta_infor.x(1);
        y_center = meta_infor.y(1); 
    end
    
    angio_struct.endpoint_data.x_center = x_center;
    angio_struct.endpoint_data.y_center = y_center;
    
    %-% Get the receive screen region for background removal
    view = imread(fullfile(file_png_folder, 'view.png'));
    view_gray = rgb2gray(im2double(view));
    receive_screen_mask = view_gray == max(view_gray(:));
    
    %-% Get the initial point of the vessel for aligning catheter endpoint
    %   and vessel start point 
    start_raw = imread(fullfile(file_png_folder, 'start.png'));
    start_image = getMaskedImage(receive_screen_mask, start_raw, ref_size)<0.5;
   
    %-% Iterate through branch related images    
    for identifier_cell = branch_identifiers
        angio_struct.segment.(identifier_cell{1}) = preprocessRhinoImage(...
            fullfile(file_png_folder, strcat(identifier_cell{1}, '.png')), ...
            receive_screen_mask, x_center, y_center, ref_size, start_image);
        angio_struct.volumn.(identifier_cell{1}) = preprocessRhinoImage(...
            fullfile(file_png_folder, strcat(identifier_cell{1}, '_contour.png')), ...
            receive_screen_mask, x_center, y_center, ref_size, start_image);
    end
    
    %-% Get the position of the stenosis
    loacation_detail_col_names = {'x_center', 'y_center'};
    loacation_detail = zeros(size(stenosis_infor,1),2);
    for iStenosis = 1:size(stenosis_infor,1)
        stnosis_raw = imread(fullfile(file_png_folder, strcat('stnosis_', num2str(iStenosis),'.png')));
        stnosis_resized = getMaskedImage(receive_screen_mask, stnosis_raw, ref_size)>0.5;  
        stnosis_image = recreateMatchedImage(x_center, y_center, ref_size, ...
            start_image, stnosis_resized);
        if sum(stnosis_image,'all')==ref_size^2
            % if there are no stenosis in the image, set to default value -1
            loacation_detail(iStenosis, 1) = -1;
            loacation_detail(iStenosis, 2) = -1;
        else
            match_boundingbox = regionprops(imcomplement(stnosis_image), 'BoundingBox').BoundingBox;
            loacation_detail(iStenosis, 1) = match_boundingbox(1)+match_boundingbox(3)/2;
            loacation_detail(iStenosis, 2) = match_boundingbox(2)+match_boundingbox(4)/2;            
        end
    end
    loacation_detail = array2table(loacation_detail, 'VariableNames', ...
        loacation_detail_col_names);
    stenosis_summary = [stenosis_detail,loacation_detail];
    angio_struct.stenosis_summary = stenosis_summary;
    
    %-% Update the stenosis summary table
    stenosis_summary_tab = [stenosis_summary_tab; stenosis_summary];
    
    %-% Load Real Background Image
    if contains(batch_id, 'UKR')
        back_image_path = fullfile(base_data_path, 'BackGround_Image', 'Clean', ...
            strcat(angio_struct.file_name,'.png'));
    else
        meta_infor = meta_infors(ismember(meta_infors.name_combine,angio_struct.file_name),:);
        % back_image_path = fullfile(meta_infor.save_dir{1}, meta_infor.background{1});
        % path_combo = string(split(meta_infor.save_dir{1},'\'));
        % base_combo = string(split(path_combo(1),':'));
        % element = [base_combo(2); path_combo(2:end-1)]';
        element = string(split(meta_infor.save_dir, '+'));
        back_ground_dir = turbo;
        for i = 1:length(element)
            back_ground_dir = fullfile(back_ground_dir, element(i));
        end
        back_image_path = fullfile(back_ground_dir, meta_infor.background{1});
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
    %-% Generate segmentation mask for major vessel and all vessels
    major_seg_image = imbinarize(imcomplement(angio_struct.segment.major));
    seg_image = zeros(ref_size, ref_size);
    for identifier = angio_struct.branch_identifiers
        seg_image_branch = imbinarize(imcomplement(angio_struct.segment.(identifier{1})));
        seg_image = seg_image | seg_image_branch;
    end
    %-% Create Folder for Output Saving
    if ~isfolder(angio_struct.output_folder); mkdir(angio_struct.output_folder);end
    %-% Save Demo image to folder
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
    %-% Write the angio_struct to folder
    if save_struct_flag
        save(fullfile(angio_struct.output_folder, 'angio_struct.mat'),'angio_struct');
    end
    %-% Write relevant information to file
    for iStenosis = 1:size(stenosis_infor,1)
        % 1. the synthetic image
        imwrite(angio_struct.synthetic_image, fullfile(angio_struct.output_folder, ...
            strcat('synthetic_', num2str(iStenosis),'.png')));
    end
    % 2. the stentosis information
    writetable(angio_struct.stenosis_summary,fullfile(angio_struct.output_folder,...
        'stenosis_infor.csv'),'Delimiter',',') 
    % 3. the segmentation mask of major and full vessel
    imwrite(seg_image, fullfile(angio_struct.output_folder, 'segmentation.png'));
    imwrite(major_seg_image, fullfile(angio_struct.output_folder, 'segmentation_major.png'));
end

writetable(stenosis_summary_tab,fullfile(output_save_dir, 'stenosis_detail.csv'),'Delimiter',',')  

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

