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
    try
        match_image((max(0,y_center-t_vessel)+1):(ref_size-abs(y_center-t_vessel)+max(0,y_center-t_vessel)+1), ...
            (max(0,x_center-s_vessel)+1):(ref_size-abs(x_center-s_vessel)+max(0,x_center-s_vessel)+1)) = cropped_image;
    catch
        % There are some dimension mismatch when 
        % t_vessel = y_center, or s_vessel = x_cnter
        [x_dim, y_dim] = size(cropped_image);
        match_image((max(0,y_center-t_vessel)+1):(x_dim+(max(0,y_center-t_vessel)+1)-1), ...
            (max(0,x_center-s_vessel)+1):(y_dim+max(0,x_center-s_vessel)+1)-1) = cropped_image;
        disp(['Warning Message: dimension mismatch handled by error catch. ', ...
            't=',num2str(t_vessel), ' , y=',num2str(y_center),...
            '; s=',num2str(s_vessel),'; x=',num2str(x_center)])
    end
    % Some time the above operation may return 512*513 size of image
    match_image = match_image(1:ref_size,1:ref_size);
end