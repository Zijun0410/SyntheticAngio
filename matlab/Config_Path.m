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

base_data_path = fullfile(turbo,'Projects','Angiogram','Data','Processed','Zijun','Synthetic');
base_code_path = fullfile(turbo,'Projects','Angiogram','Code','Zijun','SyntheticAngio');

%-% Add relevant code to MATLAB path
addpath(genpath(fullfile(base_code_path, 'matlab')));
addpath(genpath(fullfile(base_code_path, 'matlab','util')));
% Variables for Dimo
% Z:\Projects\Angiogram\Data\Processed\results\preprocessed\UKR\Full\
real_image_path = fullfile(turbo,'Projects','Angiogram','Data','Processed',...
    'results', 'preprocessed', 'UKR', 'Full');

% Some variables used in Background_Image_Preparation.m
% Z:\Datasets\Angiogram\UK\Matched\Right
base_dicom_path = fullfile(turbo,'Datasets','Angiogram','UK','Matched','Right');
