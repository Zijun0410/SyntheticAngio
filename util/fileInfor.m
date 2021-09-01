function [path_map, n, files] = fileInfor(folder, filetype)
%
% fileInfor will determine the number of files having a given suffix, 
% (ie, '.m', '.mat','.txt') that reside in 'folder'
%
% INPUTS:
%   folder      string containing name of directory to be searched
%   filetype    string indicating the suffix (ie, '.txt')
%
% OUTPUTS:
%   n           integer, number of files in 'folder' having the prescribed
%               suffix
%   path_map    an instance of a map that maps file name to its full path
%   file        a Struct that contain all the directory information

    search_term = fullfile(folder, strcat('*',filetype, '*'));
    files = dir(search_term);
    n = length(files);
    path_map = containers.Map({files(:).name}, arrayfun(@(x) fullfile(x.folder,...
        x.name), files, 'UniformOutput', false));
end 