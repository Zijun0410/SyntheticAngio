%%% Synthetic the angiogram image based on the Rhino vessel image and the
%%% real background image. 

% Setting the path etc
% Generate the infor_saver_cell which contains
%     angio_struct with the following fields
%         angio_struct.meta_data = struct();
%         angio_struct.stenosis_data = struct();
%         angio_struct.endpoint_data = struct();
%         angio_struct.segment = struct();
%         angio_struct.volumn = struct();
%         angio_struct.branch_identifiers = branch_identifiers;
%         angio_struct.ref_size = ref_size;
Load_Rhino_Image

for iVess=1:length(infor_saver_cell)
    angio_struct = infor_saver_cell{iVess};
    real_image_path = fullfile(real_image_path, angio_struct.file_name, 'frame1.png');
    back_image_path = fullfile(base_data_path, 'BackGround_Image', 'Clean', ...
        stracat(angio_struct.file_name,'.png'));
    angio_struct.background = im2double(imread(back_image_path));
    angio_struct.real_image = im2double(imread(real_image_path));
    
    gaussian_factor = 2;
    confidence_factor = 0.09;
    border_threshold = 0.5;
    adjust_factor_1 = 0.02;
    adjust_factor_2 = 100;
    for identifier = angio_struct.branch_identifiers
        volumn_image = angio_struct.volumn.(identifier);
        blur_image = imgaussfilt(volumn_image, gaussian_factor);
        vessel_confidence = (1 - blur_image)*confidence_factor;
        sythetic_image = angio_struct.background - vessel_confidence;
        sythetic_image(blur_image<border_threshold) = ...
            sythetic_image(blur_image<border_threshold)-adjust_factor_1-rand(1)/adjust_factor_2;
    end
    image_display = zeros(size(sythetic_image,1),)
    % visualize image in a horizental concat way
    montage(sythetic_image,'size',[1 3]);
end
temp = cat(3, stnosis_image, stnosis_image);
    blur_image = imgaussfilt(blur_image,3);

    vessel_confidence = (1 - blur_image)*0.09;
    nomalized_sample_frame = nomalized_sample_frame - vessel_confidence;

    nomalized_sample_frame(blur_image<0.5) = nomalized_sample_frame(blur_image<0.5) - 0.02 - rand(1)/100; 

    

%--%
image_identifier = 'example_01';
dicom_identifier = '001 (7)_R.dcm';

%--%
% To get file_path_map, num_png_files
Load_Rhino_Image
% To get sample_frame, meta_data
Load_Real_Background_Image


for iBranch=1:num_png_files
    main_branch_pre = im2double(imread(file_path_map(strcat(num2str(iBranch),'.png'))));
    main_branch_gray = rgb2gray(main_branch_pre);
    %-% Clear the border of 20 pixels
    boudary_length = 20;
    image_size = size(main_branch_gray,2);
    border = zeros(image_size,image_size);
    border(:,1:boudary_length) = 1;
    border(:,image_size-boudary_length+1:end) = 1;
    border(1:boudary_length,:) = 1;
    border(image_size-boudary_length+1:end,:) = 1;
    main_branch_gray(border==1) = 1;
    
    %[emm,rect] = imcrop(main_branch_gray);
    I2 = ones(size(main_branch_gray));
    I2(250:250+1400-1,25:25+1400-1) = main_branch_gray(200:200+1400-1,250:250+1400-1);
    % imcrop(main_branch_gray,[300,50,1200,1200])
    ref_size = size(sample_frame,2);
    blur_image = imresize(I2,[ref_size ref_size]);
    % figure;imshow(main_branch);


    blur_image = imgaussfilt(blur_image,3);

    vessel_confidence = (1 - blur_image)*0.09;
    nomalized_sample_frame = nomalized_sample_frame - vessel_confidence;

    nomalized_sample_frame(blur_image<0.5) = nomalized_sample_frame(blur_image<0.5) - 0.02 - rand(1)/100;    
end

figure;imshow(nomalized_sample_frame);









% % https://www.mathworks.com/matlabcentral/answers/565271-blur-edges-of-rectangles-in-the-image
% addpath(genpath('Z:\Projects\Angiogram\Code\Zijun\BCIL-Shared\Image_Processing'))
% gaussian2 = imgaussfilt(nomalized_sample_frame,2);
% nomalized_sample_frame(main_branch<1) = gaussian2(main_branch<1);
% 
% 
% nomalized_sample_frame  = (nomalized_sample_frame-min(nomalized_sample_frame(:)))/...
%     (max(nomalized_sample_frame(:))-min(nomalized_sample_frame(:)));
% 
% J = imnonlocalfilt2(im2double(nomalized_sample_frame), 11, 3, .01, .001);
