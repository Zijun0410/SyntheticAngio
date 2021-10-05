%%% Save the first frame (ususally within the pre-release stage) of a XCA 
%%% video as the background of synthetic XCA image. Check the quality of
%%% the first frame, only leave those good ones for XCA synthesis
%%% In the future, this function should be replace by a python deep
%%% learning inference module that can decide the catheter endpoint.

%%% Author: Zijun Gao
%%% Last Update: Oct 5th, 2021
%%% Project: SyntheticAngio

%-% 1. Configure the Path
Config_Path
Rerun_Flag = 0; 

%-% 2. Iteratively save all the first frame and metadata infor based on the
%      fields of interest
[dicom_path_map, num_dicom_files, file_struct] = fileInfor(base_dicom_path,'.dcm');
field_of_interest = {'PositionerPrimaryAngle','PositionerSecondaryAngle',...
      'DistanceSourceToDetector','DistanceSourceToPatient',...
      'EstimatedRadiographicMagnificationFactor','ExposureTime',...
      'TableHeight','DistanceObjectToTableTop','BeamAngle'};

if Rerun_Flag
    for iFile = 119:num_dicom_files
        dicom_file_name = file_struct(iFile).name;
        string_splits = string(split(dicom_file_name,'.'));
        png_save_path = fullfile(base_data_path, 'BackGround_Image','All',...
            strcat(string_splits(1), '.png'));
        meta_save_path = fullfile(base_data_path, 'Meta_Data',...
            strcat(string_splits(1), '.mat'));
        % Load the metadata
        meta_data = angioMetadata(dicom_path_map(dicom_file_name), field_of_interest);
        % Load the background frame, which is the first frame of the video
        angio_video = squeeze(dicomread(dicom_path_map(dicom_file_name))); 
        background_frame = squeeze(im2double(angio_video(:,:,1)));
        % Save all the infor
        imwrite(background_frame, png_save_path);
        save(meta_save_path, 'meta_data');
    end
end

%-% 3. Initiate a manual process to check if the quality of the first frame is Okay
if ~isfile(fullfile(base_data_path, 'BackGround_Image', 'quality_label.csv'))
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

%-% 4. Move those clean files to another folder
good_quality = quality_tab(quality_tab.quality_label==0,:);
if Rerun_Flag    
    for iFile = 1:size(good_quality,1)
        png_save_path = fullfile(base_data_path, 'BackGround_Image', 'All',...
            strcat(good_quality.file_name(iFile), '.png'));
        copyfile(png_save_path,fullfile(base_data_path, 'BackGround_Image', 'Clean'))
    end
end

%-% 5. Annotate the postion of the end point of catheter with photoshop.
% Save at Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\BackGround_Image\Annotated
annotated_image_dir = fullfile(base_data_path, 'BackGround_Image', 'Annotated');

%-% 6. Save the position of catheter endpoint in the form of center-size
%      Coordinates (center_x, center_y, width, height) on a (0,1) scale.

meta_table = [];
bounding_box = zeros(size(good_quality,1),4);
for iGood = 1:size(good_quality,1)
    %-% Read in image 
    png_file_name = strcat(good_quality.file_name{iGood},'.png.png');
    loaded_annotated = im2double(imread(fullfile(annotated_image_dir, png_file_name)));
    loaded_annotated = rgb2gray(loaded_annotated);
    annotation_binary = (loaded_annotated==1);
    annotation = bwareafilt(annotation_binary,1);
    
    %-% Obtain the location of bounding box
    annotation_boundingbox = regionprops(annotation, 'BoundingBox').BoundingBox;
    bounding_box(iGood,1) = annotation_boundingbox(1)+annotation_boundingbox(3)/2;
    bounding_box(iGood,2) = annotation_boundingbox(2)+annotation_boundingbox(4)/2;
    bounding_box(iGood,3) = annotation_boundingbox(3)+2;
    bounding_box(iGood,4) = annotation_boundingbox(4)+2;
    
    %-% Readin Metadata to form the final metadata table
    meta_save_path = fullfile(base_data_path, 'Meta_Data',...
            strcat(good_quality.file_name{iGood}, '.mat'));
    meta_data = importdata(meta_save_path);
    meta_tab = struct2table(meta_data);
    meta_table = [meta_table; meta_tab];
end

bounding_box_table = array2table(bounding_box,'VariableNames',{'CenterX', 'CenterY','Width','Height'});
file_name_table = array2table(good_quality.file_name,'VariableNames',{'FileName'});
final_table = [file_name_table, bounding_box_table, meta_table];
writetable(final_table,fullfile(base_data_path, 'meta_summary.csv'),'Delimiter',',');

%-% TODO: Get the next frame for some of the image
% need_next_frame = quality_tab(quality_tab.quality_label==3,:);