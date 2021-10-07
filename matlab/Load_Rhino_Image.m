%%% Load information from Rhnio generated images in to a data struct,
%%% the covered information includes:
%%%  1. The bounding box of stenosis region
%%%  2. The start point of vessel structure
%%%  3. The binary and grayscale mask of different branches
%%%  

%%% Author: Zijun Gao
%%% Last Update: Oct 5th 2021
%%% Project: SyntheticAngio

batch_num = 3; 
% Generated on Sep 30, 2021, 282 image groups in total
% the distance of contour set to 0.15 and the alpha value of hatch is 1

Config_Path

%-% Some default setting
image_load_dir = fullfile(base_data_path, 'Rhino_Output', num2str(batch_num));
ref_size = 512;
% meta_data_dir = fullfile(base_data_path, 'Meta_Data');
branch_identifiers = {'major', 'branch_1', 'branch_2', 'branch_3', 'branch_4', 'branch_5'};

%-% Load stenosis infor generated from Rhino-python scripts
stenosis_data = readtable(fullfile(image_load_dir, 'stnosis_infor.csv'));
stenosis_infors = sortrows(stenosis_data,1);
% TODO: modify here for future change
stenosis_infors.Properties.VariableNames = {'index', 'fileName', 'stenosis_flag', ...
    'stenosis_location', 'effect_region', 'percentage', 'distanceSourceToDetector',... 
    'distanceSourceToPatient', 'positionerPrimaryAngle', 'positionerSecondaryAngle'};

%-% Load metedata generated from dicom file and manusal annotation
  % See the matlab scripts: Background Image Preparation
meta_infors = readtable(fullfile(base_data_path, 'meta_summary.csv'));

%-% Initate the information saver
infor_saver_cell = cell(size(stenosis_infors,1),1);
%%
%-% Iterate through the files
for iCase = 1:size(stenosis_infors,1)
    %% 
    %-% Initate AngioStruct
    angio_struct = struct();
    angio_struct.meta_data = struct();
    angio_struct.stenosis_data = struct();
    angio_struct.endpoint_data = struct();
    angio_struct.segment = struct();
    angio_struct.volumn = struct();
    angio_struct.branch_identifiers = branch_identifiers;
    angio_struct.ref_size = ref_size;
    
    % Get case specifice stenosis_infor
    stenosis_infor = stenosis_infors(iCase,:);
    filename_cell = stenosis_infor.fileName;
    % TODO: modify here for the bug current bug in the stnosis_infor generation
    file_name = filename_cell{1}; 
    angio_struct.file_name = file_name;
    % stnosis_flag = ['stnosis_flag', 'stenosis_location','effect_region','percentage'];
    angio_struct.stenosis_data.stenosis_percentage = stenosis_infor.percentage;
    angio_struct.stenosis_flag = stenosis_infor.stenosis_flag;
    % TODO: check here as well
    if stenosis_infor.percentage < 0.5
        angio_struct.stenosis_data.stenosis_grade = 1;
    elseif stenosis_infor.percentage < 0.7
        angio_struct.stenosis_data.stenosis_grade = 2;
    else
        angio_struct.stenosis_data.stenosis_grade = 3;
    end
    %_% Display information when running
    disp(['Running the ', num2str(iCase), ' case. File name: ', file_name])
    
    file_png_folder = fullfile(image_load_dir, file_name);
    
    % Get the corresponding meta data and the center of catheter endpoint 
    meta_infor = meta_infors(contains(meta_infors.FileName,file_name),:);
    x_center = meta_infor.CenterX;
    y_center = meta_infor.CenterY;    
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
    stnosis_raw = imread(fullfile(file_png_folder, 'stnosis.png'));
    stnosis_resized = getMaskedImage(receive_screen_mask, stnosis_raw, ref_size)<0.5;  
    stnosis_image = recreateMatchedImage(x_center, y_center, ref_size, start_image, stnosis_resized);
    match_boundingbox = regionprops(stnosis_image, 'BoundingBox').BoundingBox;
    angio_struct.stenosis_data.x_center = match_boundingbox(1)+match_boundingbox(3)/2;
    angio_struct.stenosis_data.y_center = match_boundingbox(2)+match_boundingbox(4)/2;
    
    matched_start_image = recreateMatchedImage(x_center, y_center, ref_size, start_image, start_image);
    match_boundingbox = regionprops(matched_start_image, 'BoundingBox').BoundingBox;
    match_x_center = match_boundingbox(1)+match_boundingbox(3)/2;
    match_y_center = match_boundingbox(2)+match_boundingbox(4)/2;
    
    infor_saver_cell{iCase,1} = angio_struct;
end
