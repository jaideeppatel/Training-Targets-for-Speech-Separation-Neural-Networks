f = fullfile('J:\Box Sync\MLSP_Project\Python_project','wavefiles\irm_all_noise_male_gender_female_test');
file_struct=dir(fullfile(f,'*.mat'));
Fs = 16000;
STOI_results = containers.Map;
fnames = []
i=1;
for k=1:size(file_struct)
    fname = file_struct(k).name
    file = fullfile(f,fname);
    s = load(file);
    x = s.s_original;
    y = s.s_regenerated;
    d = stoi(x,y,Fs)
    key = strtok(fname,'.');
    STOI_results(fname) = d;
    fnames = [fnames strcat(fname,'-')];
    i=i+1;
end
% fileID = fopen('ibm_all_noise_all_gender_values.txt','w');
% fprintf(fileID,'%s \n',STOI_results.values);
% fclose(fileID);
csvwrite('irm_all_noise_male_gender_female_test_values.txt',STOI_results.values)
fileID = fopen('irm_all_noise_male_gender_female_test_keys.txt','w');
fprintf(fileID,'%s \n',fnames);
fclose(fileID);