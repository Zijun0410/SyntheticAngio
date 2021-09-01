%%% Load Real World Angiogram image frame in Pre-release Stage 
%%% Author: Zijun Gao
%%% Last Update: Ang 24th 2021
%%% Project: SyntheticAngio

meta_data = angioMetadata(fullfile(base_dicom_path, dicom_identifier));
% meta_data.PositionerPrimaryAngle
% meta_data.PositionerSecondaryAngle
% meta_data.DistanceSourceToDetector
% meta_data.DistanceSourceToPatient
angio_video = squeeze(dicomread(fullfile(base_dicom_path, dicom_identifier))); 
sample_frame = squeeze(im2double(angio_video(:,:,1)));
hi_frame = squeeze(im2double(angio_video(:,:,15)));
frame_diff = hi_frame - sample_frame;

nomalized_frame_diff  = (frame_diff-min(frame_diff(:)))/...
    (max(frame_diff(:))-min(frame_diff(:)));

border_region = borderDetection(sample_frame,meta_data);

% Remove border.
sample_frame(border_region) = NaN;

nomalized_sample_frame  = (sample_frame-min(sample_frame(:)))/...
    (max(sample_frame(:))-min(sample_frame(:)));

% % Get the (discrete) PDF with 256 bins.
% sample_frame_scalar = nomalized_sample_frame(:);
% sample_frame_scalar(isnan(sample_frame_scalar)) = [];
% [counts, bins] = imhist(sample_frame_scalar,256);
% [~, max_index] = max(counts);
% vessel_value = bins(max_index);
% 
% 
% Ipdf = counts/sum(counts(:));
% figure;plot(Ipdf);
% 
% % Determine bin locations where the median differs from the bin value by
% % more than 1% (0.01). These are spikes that occur due to preprocessing.
% % Remove them.
% medians = medfilt1(Ipdf, 5);
% diff = Ipdf - medians;
% diff(1:150) = 0; % avoid false positives at beginning of histogram
% idx = find(diff > 0.01);
