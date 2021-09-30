%%% Load information from Rhnio generated images in to a data struct,
%%% the covered information includes:
%%%  1. The bounding box of stenosis region
%%%  2. The start point of vessel structure
%%%  3. The binary and grayscale mask of different branches
%%%  

%%% Author: Zijun Gao
%%% Last Update: Sep 30th 2021
%%% Project: SyntheticAngio

%-% 
image_load_dir = fullfile(base_data_path, 'Rhino_Output', num2str(batch_num));
% meta_data_dir = fullfile(base_data_path, 'Meta_Data');
branch_identifiers = {'major', 'branch_1', 'branch_2', 'branch_3', 'branch_4', 'branch_5'};
%[file_path_map, num_png_files, ~] = fileInfor(fullfile(image_load_dir,image_identifier),'.png'); 
stnosis_infor = readtable(fullfile(image_load_dir, 'stnosis_infor.csv'));
for iCase = 1:size(stnosis_infor,1)
    filename_cell = stnosis_infor{iCase, 2};
    file_name = filename_cell{1}(1:end-1); % modify here for the bug cuurent bug in the stnosis_infor generation
    % stnosis_flag = ['stnosis_flag', 'stenosis_location','effect_region','percentage'];
    % check here as well
    stenosis_percentage = stnosis_infor{iCase, 5};
    % stenosis_label = stnosis_infor{iCase, 3};
    file_png_folder = fullfile(image_load_dir, file_name);
    for identifier_cell = branch_identifiers
        % Write a function here to crop the receive screen region
        view = imread(fullfile(file_png_folder, 'view.png'));
        view_gray = rgb2gray(im2double(view));
        receive_screen_mask = view_gray == max(view_gray(:));
        [ii,jj] = ind2sub(size(receive_screen_mask),find(receive_screen_mask>0));
        ymin=min(ii);ymax=max(ii);xmin=min(jj);xmax=max(jj);
        imCropped = imcrop(view_gray,[xmin,ymin,xmax-xmin+1,ymax-ymin]);
        % Apply on different branches
        branch_image = imread(fullfile(file_png_folder, strcat(identifier_cell{1}, '.png')));
        % Resize the image to (512, 512)
        % Find the 
    end
end