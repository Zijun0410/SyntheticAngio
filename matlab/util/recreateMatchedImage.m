function match_image = recreateMatchedImage(x_center, y_center, ref_size, ...
    vessel_start_image, target_image)
    %%% crop the inage and merge the patch back into a image so that the end
    %%% position of a catheter and the initial point of vessel matches
    x_center = ceil(x_center);
    y_center = ceil(y_center);
    match_image = zeros(ref_size, ref_size) + 1;
    annotation_boundingbox = regionprops(vessel_start_image, 'BoundingBox').BoundingBox;
    s_vessel = floor(annotation_boundingbox(1)+annotation_boundingbox(3)/2);
    t_vessel = floor(annotation_boundingbox(2)+annotation_boundingbox(4)/2);
    cropped_image = imcrop(target_image,[max(max(s_vessel-x_center,0),1), ...
        max(max(t_vessel-y_center,0),1), ref_size-abs(x_center-s_vessel), ref_size-abs(y_center-t_vessel)]);
    match_image((max(0,y_center-t_vessel)+1):(ref_size-abs(y_center-t_vessel)+max(0,y_center-t_vessel)+1), ...
        (max(0,x_center-s_vessel)+1):(ref_size-abs(x_center-s_vessel)+max(0,x_center-s_vessel)+1)) = cropped_image;
    % Some time the above operation may return 512*513 size of image
    match_image = match_image(1:ref_size,1:ref_size);
end