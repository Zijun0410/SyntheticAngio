%%% Load Rhnio generated images
%%% Author: Zijun Gao
%%% Last Update: Ang 24th 2021
%%% Project: SyntheticAngio

%-% 

image_load_dir = fullfile(base_data_path, 'Sample_Image');
meta_data_dir = fullfile(base_data_path, 'Meta_Data');

[file_path_map, num_png_files, ~] = fileInfor(fullfile(image_load_dir,image_identifier),'.png'); 

