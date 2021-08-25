function metadata = angioMetadata(dicom_path)

    % Load metadata from the dicom file
    metadata = load_metadata(dicom_path);

    % Validate the values given for primary and secondary angulations. 
    metadata = validate_angulations(metadata);
end


function metadata = load_metadata(dicom_path)
    if ~isempty(dicom_path)
        metadata = dicominfo(dicom_path);
    else
        warning('No metadata loaded.')
        metadata = struct();
    end
end


function metadata = validate_angulations(metadata)
% Set defaults for the LAO/RAO and CRA/CAU angulation if they
% are empty in the metadata structure.
    if isfield(metadata, 'PositionerPrimaryAngle')
        if isempty(metadata.PositionerPrimaryAngle)
            metadata.PositionerPrimaryAngle = 0;
        end
    else
        warning('No LAO/RAO angulation loaded.')
        metadata.PositionerPrimaryAngle = 0;
    end
    if isfield(metadata, 'PositionerSecondaryAngle')
        if isempty(metadata.PositionerSecondaryAngle)
            metadata.PositionerSecondaryAngle = 0;
        end
    else
        warning('No CRA/CAU angulation loaded.')
        metadata.PositionerSecondaryAngle = 0;
    end
end