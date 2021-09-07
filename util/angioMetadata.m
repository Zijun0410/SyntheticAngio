function meta_data_out = angioMetadata(dicom_path, field_of_interest)
    %%% Given the dicom path, return a metadata struct that contain fields or
    %%% interest, if a field does not exist, mark as -1.
    %%% Author: Zijun Gao
    %%% Last Update: Sep 7th, 2021
    %%% Project: SyntheticAngio
    
    % Load metadata from the dicom file
    meta_data = load_metadata(dicom_path);
    % Validate the values given for primary and secondary angulations. 
    meta_data_out = validate_fields(meta_data, field_of_interest);
end

function metadata = load_metadata(dicom_path)
    if ~isempty(dicom_path)
        metadata = dicominfo(dicom_path);
    else
        warning('No metadata loaded.')
        metadata = struct();
    end
end

function meta_data_out = validate_fields(meta_data, field_of_interest)
% Set value for field of interest with a default of -1.
    meta_data_out = struct();
    for field_cell = field_of_interest
        % Iterate through field of interest
        target_field = string(field_cell);
        if isfield(meta_data,target_field)
            meta_data_out.(target_field) = meta_data.(target_field);
        else
            meta_data_out.(target_field) = -1;
        end
    end
end