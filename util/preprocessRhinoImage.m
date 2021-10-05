function image_processed = preprocessRhinoImage(fullpath, receive_screen_mask, ...
    x_center, y_center, ref_size, start_image)
    %%% 1. Read in the raw image
    %%% 2. Remove irrelevant border and resize image
    %%% 3. Match the start point of the vessel(from rhino image) and the end
    %%%    point of catheter(from real bakcground image/metadata)
    image_raw = imread(fullpath);
    image_resized = getMaskedImage(receive_screen_mask, image_raw, ref_size);
    image_processed = recreateMatchedImage(x_center, y_center, ref_size, start_image, image_resized);
end