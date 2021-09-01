%%% Save the first frame (ususally within the pre-release stage) of a XCA 
%%% video as the background of synthetic XCA image. Check the quality of
%%% the first frame, only leave those good ones for 
%%% Author: Zijun Gao
%%% Last Update: Ang 31th 2021
%%% Project: SyntheticAngio

Config_Path
[dicom_path_map, num_dicom_files, file_struct] = fileInfor(base_dicom_path,'.dcm'); 

for iFile = 1:num_dicom_files
    dicom_file_name = file_struct(iFile).name;
    string_splits = string(split(dicom_file_name,'.'));
    png_save_path = fullfile(base_data_path, 'BackGround_Image','All',...
        strcat(string_splits(1), '.png'));
    meta_save_path = fullfile(base_data_path, 'Meta_Data',...
        strcat(string_splits(1), '.mat'));
    % Load the metadata
    meta_data = angioMetadata(dicom_path_map(dicom_file_name));
    % Load the background frame, which is the first frame of the video
    angio_video = squeeze(dicomread(dicom_path_map(dicom_file_name))); 
    background_frame = squeeze(im2double(angio_video(:,:,1)));
    
    imwrite(background_frame, png_save_path);
    save(meta_save_path, 'meta_data');
    
end

%% A manual process to check if the quality of the image is Okay
if isfile(fullfile(base_data_path, 'BackGround_Image', 'quality_label.csv'))
    quality_label = zeros(1,num_dicom_files);
    file_name = cell(num_dicom_files,1);   
    for iFile = 1:num_dicom_files
        dicom_file_name = file_struct(iFile).name;
        string_splits = string(split(dicom_file_name,'.'));
        file_name{iFile} = string_splits(1);
        png_save_path = fullfile(base_data_path, 'BackGround_Image','All',...
            strcat(string_splits(1), '.png'));
        background_frame = imread(png_save_path);
        figure;imshow(im2double(background_frame));
        prompt = strcat(['Frame: ', num2str(iFile), 'with name', ...
            convertStringsToChars(string_splits(1)), ', quality: ']);
        str = input(prompt,'s');
        if ~isempty(str)
            if strcmp(str, 'a')
                % some other thing there, discarded
                quality_label(iFile) = 2;
            elseif strcmp(str, 'd')  
                % Need the nexy frame
                quality_label(iFile) = 3;
            elseif strcmp(str, 'w')
                % Nothing to identify
                quality_label(iFile) = 4;
            end
        end
        close
    end

    index = 1:num_dicom_files;
    quality_tab = [array2table([index', quality_label']),cell2table(file_name)];
    writetable(quality_tab,fullfile(base_data_path, 'BackGround_Image', 'quality_label.csv'),'Delimiter',',');
else
    quality_tab = readtable(fullfile(base_data_path, 'BackGround_Image', 'quality_label.csv'));
end

%% Move those clean files to another folder
good_quality = quality_tab(quality_tab.quality_label==0,:);

for iFile = 1:size(good_quality,1)
    png_save_path = fullfile(base_data_path, 'BackGround_Image', 'All',...
        strcat(good_quality.file_name(iFile), '.png'));
    copyfile(png_save_path,fullfile(base_data_path, 'BackGround_Image', 'Clean'))
end
%% Annotate the postion of the end point of catheter with photoshop
% Save at Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\BackGround_Image\Annotated

%% TODO: Get the next frame for some of the image
need_next_frame = quality_tab(quality_tab.quality_label==3,:);