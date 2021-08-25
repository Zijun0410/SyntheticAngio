%%% Configure the default path setting under different platform
%%% Author: Zijun Gao
%%% Last Update: Ang 24th 2021
%%% Project: SyntheticAngio

%-% Define the default turbo drive path
if contains(computer,'WIN')
    turbo = 'Z:\';
else
    turbo = '/nfs/turbo/med-kayvan-lab/';
end

%-% Add relevant code to MATLAB path
current_folder = pwd;
addpath(current_folder)
addpath(genpath(fullfile(current_folder, 'util')));

base_data_path = fullfile(turbo,'Projects','Angiogram','Data','Processed','Zijun','Synthetic');
base_code_path = fullfile(turbo,'Users','zijung','Code','SyntheticAngio');
% Z:\Datasets\Angiogram\UK\Matched\Right
base_dicom_path = fullfile(turbo,'Datasets','Angiogram','UK','Matched','Right');