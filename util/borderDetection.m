function [exclusion,flag] = borderDetection(image, metadata)

if ~exist('metadata','var')
    metadata=struct();
end

% Flag if no medatada is provided.
flag = 0;

% Determine which version we shall run.
fields = string(fieldnames(metadata));

%% SHUTTER REMOVAL

% Determine the shutter shape. 
if any(fields=='ShutterShape')
    if strcmp(metadata.ShutterShape,'CIRCULAR')
        exclusion = Case1(image, metadata);
    elseif strcmp(metadata.ShutterShape,'RECTANGULAR\CIRCULAR')
        exclusion = Case2(image, metadata);
    elseif strcmp(metadata.ShutterShape,'RECTANGULAR')
        exclusion = Case3(image, metadata);
    else % Create new case based on the metadata tags if this occurs.
        exclusion = NoMetadata(image);
        flag = 1 ;
        disp(metadata.ShutterShape)
    end 
else%if input_image == 1 % If metadata is unavailable but an img is provided.
    flag = 1 ;
    exclusion = NoMetadata(image);
% else
%     exclusion = zeros(size(image));
end

%% COLLIMATOR REMOVAL

% Determine the collimator 
if any(fields=='CollimatorShape')
    if strcmp(metadata.CollimatorShape,'RECTANGULAR')
        coll_exclusion = CollimatorRectangular(image,metadata);
    else % Create new case based on metadata tags if this occurs.
        disp(metadata.CollimatorShape)
        % Do nothing. (For now.)
    end 
else 
    coll_exclusion = zeros(size(image));
    % Do nothing. (For now.)
end

%% Combine.
exclusion = exclusion | coll_exclusion;
end

%% The case of a CIRCULAR shutter.
function exclusion = Case1(image, metadata)

% Get shutter edge stats.
shutter_center = metadata.CenterOfCircularShutter;
shutter_radius = metadata.RadiusOfCircularShutter;
    
% Init
exclusion = zeros(size(image));

% Put in the circular frame.
circumference_shutter = floor(pi * shutter_radius^2);
for y = 1:circumference_shutter
    x1 = ( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) + shutter_center(2);
    x2 = -( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) + shutter_center(2);
    x3 = ( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) - shutter_center(2);
    x4 = -( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) - shutter_center(2);
    x1 = floor(x1); x2 = floor(x2); x3 = floor(x3); x4 = floor(x4);
    
    if x1 <= size(image,2) && y <= size(image,1) && x1 > 0 && y > 0
        exclusion(y,x1)=1;
    end  
    if x2 <= size(image,2) && y <= size(image,1) && x2 > 0 && y > 0
        exclusion(y,x2)=1;
    end  

end

% Fill mask regions outside of the circle.
exclusion = imdilate(exclusion,strel('disk',2,4));
exclusion = imfill(logical(exclusion), [1 1], 8);
exclusion = imfill(exclusion, [1 size(image,2)], 8);
exclusion = imfill(exclusion, [size(image,1) size(image,2)], 8);
exclusion = imfill(exclusion, [size(image,1) 1], 8);

end

%% The case of a RECTANGULAR\CIRCULAR shutter.
function exclusion = Case2(image, metadata)
    
% Get shutter edge stats.
left_vertical_edge = max(metadata.ShutterLeftVerticalEdge, 1);
right_vertical_edge = min(metadata.ShutterRightVerticalEdge, size(image,2));
upper_horizontal_edge = max(metadata.ShutterUpperHorizontalEdge, 1);
lower_horizontal_edge = min(metadata.ShutterLowerHorizontalEdge, size(image,1));
shutter_center = metadata.CenterOfCircularShutter;
shutter_radius = metadata.RadiusOfCircularShutter;
    
% Put in the rectangular frame.
exclusion = zeros(size(image));
exclusion(:,1:left_vertical_edge)=1;    
exclusion(:,right_vertical_edge:end)=1;
exclusion(1:upper_horizontal_edge,:)=1;
exclusion(lower_horizontal_edge:end,:)=1;
   
% Put in the circular frame.
circ_mask = zeros(size(image));
circumference_shutter = floor(pi * shutter_radius^2);
for y = 1:circumference_shutter
    x1 = ( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) + shutter_center(2);
    x2 = -( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) + shutter_center(2);
    x3 = ( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) - shutter_center(2);
    x4 = -( shutter_radius^2 - (y-shutter_center(1))^2 )^(1/2) - shutter_center(2);
    x1 = floor(x1); x2 = floor(x2); x3 = floor(x3); x4 = floor(x4);
    
    if x1 <= size(image,2) && y <= size(image,1) && x1 > 0 && y > 0
        circ_mask(y,x1)=1;
    end  
    if x2 <= size(image,2) && y <= size(image,1) && x2 > 0 && y > 0
        circ_mask(y,x2)=1;
    end  

end

% Fill mask regions outside of the circle.
circ_mask = imdilate(circ_mask,strel('disk',2,4));
circ_mask = imfill(logical(circ_mask), [1 1], 8);
circ_mask = imfill(circ_mask, [1 size(image,2)], 8);
circ_mask = imfill(circ_mask, [size(image,1) size(image,2)], 8);
circ_mask = imfill(circ_mask, [size(image,1) 1], 8);


% Add in the rectangular frame.
exclusion = exclusion | circ_mask;

end

%% The case of a RECTANGULAR shutter.
function exclusion = Case3(image, metadata)
    
% Get shutter edge stats.
left_vertical_edge = max(metadata.ShutterLeftVerticalEdge, 1);
right_vertical_edge = min(metadata.ShutterRightVerticalEdge, size(image,2));
upper_horizontal_edge = max(metadata.ShutterUpperHorizontalEdge, 1);
lower_horizontal_edge = min(metadata.ShutterLowerHorizontalEdge, size(image,1));

% Put in the rectangular frame.
exclusion = zeros(size(image));
exclusion(:,1:left_vertical_edge)=1;    
exclusion(:,right_vertical_edge:end)=1;
exclusion(1:upper_horizontal_edge,:)=1;
exclusion(lower_horizontal_edge:end,:)=1;

end
    
%% The case where no metadata is available. 
% Use mean/std plus a convex hull
% to estimate what the shutter frame is. 
function exclusion = NoMetadata(image)

[numrows, numcols] = size(image);

% PREVIOUS METHOD
%image = im2double(localcontrast(im2uint8(image),.99,.6));
%immax = max(image(:));
%[row,col]= find(image==immax);

% NEW METHOD 2018-10-10
% Handles the case where the background is NOT dark. Seen in the JH
% dataset. No predictible pixel intensity range for the background/edges of
% image.
bg_intensity = image(1,1);

% REMOVED 2018-10-10
% image = im2double(localcontrast(im2uint8(image),.99,.6));
%immax = max(image(:)-bg_intensity);
%[row,col]= find(image==immax+bg_intensity);
%
% vcount = 0;
% hcount = 0;
% exclusion = zeros(size(image));
% if ~isempty(find(row==1,1)) 
%      %image(1:5,100:numcols-100) = bg_intensity; % UPDATE 2018-10-10
%      %exclusion(1:5,100:numcols-100) = 1; % UPDATE 2018-10-10
%      exclusion(1:A,B:numcols-B) = 1;%ADDED 2018-10-10
%      hcount = hcount + 1;
% end
% if ~isempty(find(row==numrows,1)) 
%     %image(numrows-5:numrows,100:numcols-100) = bg_intensity; % UPDATE 2018-10-10
%     %exclusion(numrows-5:numrows,100:numcols-100) = 1;% UPDATE 2018-10-10
%     exclusion(numrows-A:numrows,B:numcols-B) = 1;% ADDED 2018-10-10
%     hcount = hcount + 1;
% end
% if ~isempty(find(col==1,1)) 
%     %image(100:numrows-100,1:5) = bg_intensity; % UPDATE 2018-10-10
%     %exclusion(100:numrows-100,1:5) = 1;% UPDATE 2018-10-10
%     exclusion(B:numrows-B,1:A) = 1;% ADDED 2018-10-10
%     vcount = vcount + 1;
% end
% if ~isempty(find(col==numcols,1))
%     %image(100:numrows-100,numcols-5:numcols) = 1; % UPDATE 2018-10-10
%     exclusion(100:numrows-100,numcols-5:numcols) = 1;% UPDATE 2018-10-10
%     exclusion(B:numrows-B,numcols-A:numcols) = 1;% ADDED 2018-10-10
%     vcount = vcount + 1;
% end
% 
% if vcount > 0 && hcount > 0
%     %exclusion = zeros(size(image)); % UPDATE 2018-10-10
%     return
% end

t = bwareaopen((abs(image-bg_intensity)<0.001),50);% ADDED 2018-10-10
image(t==1)=0;% UPDATE 2018-10-10

m = nanmean(image(:));
s = nanstd(image(:));

% exclusion = imbinarize(image/max(image(:)),max(m-s,.2));% UPDATE 2018-10-10
if s < 0.2
    exclusion = imbinarize(image/max(image(:)),m-s);
else
    exclusion = imbinarize(image/max(image(:)),m-.5*s);
end

% exclusion1 = imbinarize(dn1-min(dn1(:)),0.2);
exclusion = ~imfill(exclusion,'holes');
exclusion = imdilate(exclusion,strel('disk',3,4));
exclusion = ~bwconvhull(~exclusion);

% Exclude all border zones. % ADDED 2018-10-10
t=ones(size(exclusion));
A = ceil(numrows/100); 
t(A:numrows-A,A:numcols-A)=0;
exclusion = exclusion | t;
end

%% The rectangular collimator case.
function exclusion = CollimatorRectangular(image, metadata)

% Get delimiters.
left_vertical = metadata.CollimatorLeftVerticalEdge;
right_vertical = metadata.CollimatorRightVerticalEdge;
upper_horizontal = metadata.CollimatorUpperHorizontalEdge;
lower_horizontal = metadata.CollimatorLowerHorizontalEdge;

% Put in the rectangular frame.
exclusion = zeros(size(image));
exclusion(:,1:left_vertical)=1;    
exclusion(:,right_vertical:end)=1;
exclusion(1:upper_horizontal,:)=1;
exclusion(lower_horizontal:end,:)=1;

end
