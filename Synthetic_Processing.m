%%% Synthetic the angiogram image

Config_Path

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
    main_branch = imresize(I2,[ref_size ref_size]);
    % figure;imshow(main_branch);


    main_branch = imgaussfilt(main_branch,3);

    vessel_confidence = (1 - main_branch)*0.09;
    nomalized_sample_frame = nomalized_sample_frame - vessel_confidence;

    nomalized_sample_frame(main_branch<0.5) = nomalized_sample_frame(main_branch<0.5) - 0.02 - rand(1)/100;    
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
