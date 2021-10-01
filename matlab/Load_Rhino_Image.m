%%% Load information from Rhnio generated images in to a data struct,
%%% the covered information includes:
%%%  1. The bounding box of stenosis region
%%%  2. The start point of vessel structure
%%%  3. The binary and grayscale mask of different branches
%%%  

%%% Author: Zijun Gao
%%% Last Update: Sep 30th 2021
%%% Project: SyntheticAngio

ref_size = 512;
%-% 
image_load_dir = fullfile(base_data_path, 'Rhino_Output', num2str(batch_num));
% meta_data_dir = fullfile(base_data_path, 'Meta_Data');
branch_identifiers = {'major', 'branch_1', 'branch_2', 'branch_3', 'branch_4', 'branch_5'};
%[file_path_map, num_png_files, ~] = fileInfor(fullfile(image_load_dir,image_identifier),'.png'); 
stnosis_infor = readtable(fullfile(image_load_dir, 'stnosis_infor.csv'));
for iCase = 1:size(stnosis_infor,1)
    %% 
    filename_cell = stnosis_infor{iCase, 2};
    file_name = filename_cell{1}(1:end-1); % modify here for the bug current bug in the stnosis_infor generation
    % stnosis_flag = ['stnosis_flag', 'stenosis_location','effect_region','percentage'];
    % check here as well
    stenosis_percentage = stnosis_infor{iCase, 5};
    % stenosis_label = stnosis_infor{iCase, 3};
    file_png_folder = fullfile(image_load_dir, file_name);
    
    %-% Get the receive screen region for background removal
    view = imread(fullfile(file_png_folder, 'view.png'));
    view_gray = rgb2gray(im2double(view));
    receive_screen_mask = view_gray == max(view_gray(:));
    %-% Get the initial point of the vessel for catheter endpoint
    %   and vessel start point alignment
    start_raw = imread(fullfile(file_png_folder, 'start.png'));
    start_image = getMaskedImage(receive_screen_mask, start_raw, ref_size)<0.5;
    x_center = 209.5;
    y_center = 119.5;       
    %-% Iterate through branch related images    
    for identifier_cell = branch_identifiers
        branch_raw = imread(fullfile(file_png_folder, strcat(identifier_cell{1}, '.png')));
        hatch_raw = imread(fullfile(file_png_folder, strcat(identifier_cell{1}, '_contour.png')));
        branch_resized = getMaskedImage(receive_screen_mask, branch_raw, ref_size);
        branch_image = recreateMatchedImage(x_center, y_center, ref_size, start_image, branch_resized);
        hatch_resized = getMaskedImage(receive_screen_mask, hatch_raw, ref_size);
        hatch_image = recreateMatchedImage(x_center, y_center, ref_size, start_image, hatch_resized);
    end
    %-% Get the position of the stenosis
    stnosis_raw = imread(fullfile(file_png_folder, 'stnosis.png'));
    stnosis_resized = getMaskedImage(receive_screen_mask, stnosis_raw, ref_size)<0.5;  
    stnosis_image = recreateMatchedImage(x_center, y_center, ref_size, start_image, stnosis_resized);
    
end
