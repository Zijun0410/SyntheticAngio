function resized_image = getMaskedImage(mask, raw_image, ref_size)
% Turn the raw image into double and leave only one channel
% Crop the image with the mask, where the region to be kept are labeled 1
% Resize to (ref_size, ref_size)
    gray_color_image = rgb2gray(im2double(raw_image));
    [ii,jj] = ind2sub(size(mask),find(mask>0));
    ymin=min(ii);ymax=max(ii);xmin=min(jj);xmax=max(jj);
    % Make sure it's a square cut by limiting the dimensions
    dimension = min(xmax-xmin+1, ymax-ymin+1);
    cropped_image = imcrop(gray_color_image,[xmin,ymin,dimension,dimension]);
    resized_image = imresize(cropped_image,[ref_size ref_size]);
end