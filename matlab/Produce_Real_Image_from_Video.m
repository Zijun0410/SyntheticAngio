%% Add the correct path to Matlab working directory

if contains(computer,'WIN')
    turbo = 'Z:\';
else
    turbo = '/nfs/turbo/med-kayvan-lab/';
end

%%%%%%%%%%%%%%%%%%%%%%%%%% CUSTOMIZE THESE LINES %%%%%%%%%%%%%%%%%%%%%%%%%%
code_path = fullfile(turbo,'Projects','Angiogram','Code','Zijun','AngioAid');
toolkit_path = fullfile(turbo,'Projects','Angiogram','Code','Zijun',...
    'BCIL-Shared','Image_Processing');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cd(code_path)
addpath(genpath(code_path))
addpath(genpath(toolkit_path))

%% Set folder infor
path_prefix = fullfile("Projects", "Angiogram", "Data", "Processed", ...
    "Zijun", "Synthetic");
save_folder = "Real_Image";
load_folder = "";
% db_id = 'UM';
db_id = 'UK';
side = 'R';

warning('off','all')
% Create The database object
db = Database(turbo, path_prefix, db_id, side, save_folder, load_folder);

%% Iterate through all the object in the database
count = 1;
saving_struct = struct();
saving_struct.index = zeros(db.size*3, 1);
saving_struct.save_dir = strings(db.size*3, 1);
for i = 1:db.size
    
    %%
    video_name = db.names{i};
    if endsWith(video_name,'.dcm')
        video_name = strrep(video_name,'.dcm','');
    end
%     names = split(vidNames,"_");
%     video_name = names(:,1);
%     unique_cases = unique(video_name);
    
    disp(['Running the ', num2str(i), ' case. File name: ', video_name])
        
    try
        angio_obj = Angiogram(video_name, turbo, path_prefix, db_id, side, ...
            save_folder, load_folder);
        for iFrame = 1:3
            % mkdir(fullfile(angio_obj.savefolder, angio_obj.name))
            save_path = fullfile(angio_obj.savedir, angio_obj.name, ...
                strcat('frame', num2str(iFrame), '.png'));
            % imwrite(angio_obj.frame(:,:,iFrame), save_path);
            saving_struct.index(count) = count;
            saving_struct.save_dir(count) = save_path;
            count = count+1;
        end
    catch
        disp(['Error: in the ', num2str(i), ' case.'])
    end
end

saving_struct.save_dir = replace(saving_struct.save_dir,'\','+');

saving_table = struct2table(saving_struct);
writetable(saving_table,fullfile(angio_obj.savefolder, 'image_infor.csv'),'Delimiter',',')