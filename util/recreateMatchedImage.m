function match_image = recreateMatchedImage(x_center, y_center, ref_size, ...
    vessel_start_image, target_image)
    match_image = zeros(ref_size, ref_size) + 1;
    annotation_boundingbox = regionprops(vessel_start_image, 'BoundingBox').BoundingBox;
    s_vessel = annotation_boundingbox(1)+annotation_boundingbox(3)/2;
    t_vessel = annotation_boundingbox(2)+annotation_boundingbox(4)/2;
    cropped_image = imcrop(target_image,[max(max(x_center-s_vessel,0),1), ...
        max(max(y_center-t_vessel,0),1), ref_size-abs(x_center-s_vessel), ref_size-abs(y_center-t_vessel)]);
    match_image((max(0,y_center-t_vessel)+1):(ref_size-abs(y_center-t_vessel)+max(0,y_center-t_vessel)+1), ...
        (max(0,x_center-s_vessel)+1):(ref_size-abs(x_center-s_vessel)+max(0,x_center-s_vessel)+1)) = cropped_image;
end