% Description:  This code is used to process dataset 3.
%               
% Author:       Minh Tran Duc Nguyen
%               
% Date:         October, 2022
%% ================================================================= %%
clear all;
clc;
warning off
close all;
%% ================================================================= %%
%% Each individual measurement of a subject========================= %%
    %% CONFIG
fs = 200;
FOLD=10;
    %% IMPORTING DATA INTO MATLAB
path=[uigetdir,'\'];
checklist_FileName = dir(path);
ListFileName = {};
% Take all .mat file out of folder
for i = 1:length(checklist_FileName)
    if contains(checklist_FileName(i).name,'.mat')
        ListFileName{end+1} = checklist_FileName(i).name;
    end
end
clear checklist_FileName
for num = 1 : length(ListFileName)
    % Import .mat file
    FileName = char(ListFileName(num));
    rawdata = importdata([path,FileName]);
    for i = 1:size(rawdata.data,1)
        EEG{i} = rawdata.data(i,:,:);
        EEG{i} = squeeze(EEG{i})';
    end

        %% PREPROCESSING
%         In this section, choose one of the time segments to run by
%         removing the "%" symbol.
            %% 1-s segment
%     for i = 1:length(EEG)
%         EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%         for j = 1:round(length(EEG_4s))/fs
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:j*fs)', fs, [4 40])';
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:j*fs)', fs, [8 14])';
%             
%             data_4class{i,j} = data_temp;
%             label_4class(i,j) = (rawdata.label(1,i) + 1);
%         end
%     end
%     data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
%     label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

            %% 1-second overlapping 0.5-second
%     for i = 1:length(EEG)
%         EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%         for j = 1:7
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(EEG_4s(:,(j-1)*(fs-100)+1:j*fs-(j-1)*(fs-100))', fs, [4 40])';
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(EEG_4s(:,(j-1)*(fs-100)+1:j*fs-(j-1)*(fs-100))', fs, [8 14])';
%             
%             data_4class{i,j} = data_temp;
%             label_4class(i,j) = (rawdata.label(1,i) + 1);
%         end
%     end
%     data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
%     label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

            %% 2-second overlapping 1-second
%     for i = 1:length(EEG)
%         EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%         for j = 1:3
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:(j+1)*fs)', fs, [4 40])';
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:(j+1)*fs)', fs, [8 14])';
%             
%             data_4class{i,j} = data_temp;
%             label_4class(i,j) = (rawdata.label(1,i) + 1);
%         end
%     end
%     data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
%     label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

            %% Whole segment
    for i = 1:length(EEG)
                %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_4class{i} = eeg_filter(EEG{i}(:,:)', fs,[4 40])';
                %% Case 02: Apply feature selections: CSP
        data_4class{i} = eeg_filter(EEG{i}(:,:)', fs,[8 14])';
        
        data_4class{i} = data_4class{i}(:,201:1000);%only take 4-second MI
    end
    data_4class = data_4class'; 
    label_4class = (rawdata.label + 1)';

    fprintf('Preprocessing Done\n');

    %% 4-CLASS OVA - KFOLD - FEATURE EXTRACTION - CLASSIFICATISON
    %% 4-CLASS OVA - KFOLD
    class=[1; 2; 3; 4];     %(1)RH  (2)LH  (3)RF  (4)LF
    fprintf('OVA 4-class Classification (%d fold-cross validation)\n',FOLD);
    CV= cvpartition(label_4class,'Kfold',FOLD);
    cfm_LDA=[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];
    acc=[];
    for fold=1:CV.NumTestSets
        tic
        clear data label data_train data_test label_train label_test y_final
        data_train = data_4class(CV.training(fold)==1,:);  
        data_test = data_4class(CV.test(fold)==1,:); 
        label_train = label_4class(CV.training(fold)==1);
        label_test = label_4class(CV.test(fold)==1);

        for i=1:size(class,1)
            clear epoch_train epoch_test y_train x_train x_test score
            epoch_train = data_train;
            epoch_test = data_test;
            y_train = label_train;
            y_train(find(y_train~=class(i)))=5;
            y_train(find(y_train==class(i)))=1;
            y_train(find(y_train==5))=2;
            
            %% FEATURE EXTRACTION
%             In this block, choose a feature extraction method to process
%             the data. The way to choose is to remove the "%" symbol in
%             that method code.
                %% CSP
           trainParams.m = 3; % 2m <= channels
           [WCSP, x_train, x_test] = CSP_training(epoch_train, y_train, epoch_test, trainParams);
           band_selected(fold)={'None'};
           SaveFileName='CSP';

                %% FBCSP 
%            params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%                 20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%            params.fs=fs;
%            params.m = 3;
%            [x_train, x_test]=FBCSP_training(epoch_train, y_train, epoch_test, params);
%            band_selected(fold)={'None'};
%            SaveFileName='FBCSP';

                %% DFBCSP Fisher
%            params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%                          20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%            params.fs=fs;
%            params.m = 3;
%            [x_train, x_test, bands_selected] = DFBCSP_training_Fisher(epoch_train, y_train, epoch_test, params);
%            band_selected(fold)= {bands_selected};
%            SaveFileName='DFBCSP_Fisher';

                %% DFBCSP mRmR
%            params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%                   20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%            params.fs=fs;
%            params.m = 3;
%            [x_train, x_test, bands_selected] = DFBCSP_training_mRmR(epoch_train, y_train, epoch_test, params);
%            band_selected(fold)= {bands_selected};
%            SaveFileName='DFBCSP_mRmR';

                %% DFBCSP FmRmR
%            params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%                   20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%            params.fs=fs;
%            params.m = 3;
%            [x_train, x_test, bands_selected] = DFBCSP_training_FmRmR(epoch_train, y_train, epoch_test, params);
%            band_selected(fold)={bands_selected};
%            SaveFileName='DFBCSP_FmRmR';

           %% CLASSIFICATION
           switch i
               case 1
                model1 = fitcdiscr(x_train,y_train);
                [~,score1] = predict(model1, x_test);
               case 2
                model2 = fitcdiscr(x_train,y_train);
                [~,score2] = predict(model2, x_test);
               case 3
                model3 = fitcdiscr(x_train,y_train);
                [~,score3] = predict(model3, x_test);
               case 4
                model4 = fitcdiscr(x_train,y_train);
                [~,score4] = predict(model4, x_test);
           end
    %        bands_pair(i) = {bands_selected};
    %        WCSP_pair(i) = {WCSP};
        end

        score = [score1(:,1) score2(:,1) score3(:,1) score4(:,1)];
        %% Voting method
        for j = 1:size(score,1)
            [~,y_final(j,:)] = max(score(j,:));
        end

        A11 =  sum((y_final==1) & (label_test == 1));
        A12 =  sum((y_final==1) & (label_test == 2));
        A13 =  sum((y_final==1) & (label_test == 3));
        A14 =  sum((y_final==1) & (label_test == 4));
        A21 =  sum((y_final==2) & (label_test == 1));
        A22 =  sum((y_final==2) & (label_test == 2));
        A23 =  sum((y_final==2) & (label_test == 3));
        A24 =  sum((y_final==2) & (label_test == 4));
        A31 =  sum((y_final==3) & (label_test == 1));
        A32 =  sum((y_final==3) & (label_test == 2));
        A33 =  sum((y_final==3) & (label_test == 3));
        A34 =  sum((y_final==3) & (label_test == 4));
        A41 =  sum((y_final==4) & (label_test == 1));
        A42 =  sum((y_final==4) & (label_test == 2));
        A43 =  sum((y_final==4) & (label_test == 3));
        A44 =  sum((y_final==4) & (label_test == 4));
        cfm = [A11 A12 A13 A14; A21 A22 A23 A24; A31 A32 A33 A34; A41 A42 A43 A44]; 
        cfm_LDA = cfm_LDA + cfm; 

        cfm_result(fold)={cfm};
        BandSelected(fold)={band_selected};
        miniTime(fold)={toc}; 
        varName(fold)=cellstr(sprintf('FOLD %d',fold));
        
        % clear variables -- FIXED
        clear bands_pair WCSP_pair y_final
    end
%     result.cfm{i}=cfm_LDA;
    %accuracy
    acc_cfm = 100*trace(cfm_LDA)/sum(cfm_LDA(:));
    %F1-score
    precision=cfm_LDA(1,1)/sum(cfm_LDA(1,:));
    recall=cfm_LDA(1,1)/sum(cfm_LDA(:,1));
    F1=(2 * precision * recall) / (precision + recall);
    %display
    fprintf('accuracy= %.1f; ',acc_cfm);
    fprintf('F1-score= %.4f \n',F1);
    disp(cfm_LDA);
    %% Save result into workspace
        % (FileName(1:end-4)) = 'TR08_429_raw' => remove '.mat' -- FIXED
    table=cell2table([miniTime; cfm_result; band_selected],...
    'VariableNames',varName','RowNames',{'Time','cfm','Band selected'});
    final_result.(FileName(1:end-4)).cfm_LDA=cfm_LDA;
    final_result.(FileName(1:end-4)).acc=round(acc_cfm,2);
    final_result.(FileName(1:end-4)).F1=round(F1,2);
    final_result.(FileName(1:end-4)).Table=table;
    
    % clear variable
    clear EEG data_4class label_4class band_selected cfm_result...
        BandSelected miniTime varName
end

%% ================================================================= %%
%% All individual measurement of a subject========================== %%
clearvars -except path fs FOLD ListFileName final_result
%% IMPORTING DATA INTO MATLAB
EEG={}; LABEL=[];
for num = 1 : length(ListFileName)
    % Import .mat file
    FileName = char(ListFileName(num));
    rawdata = importdata([path,FileName]);
    for i = 1:size(rawdata.data,1)
        temp_EEG{i} = rawdata.data(i,:,:);
        temp_EEG{i} = squeeze(temp_EEG{i})';
        
        % Add 'rawdata.label' into 'LABEL'
        LABEL(end+1) = rawdata.label(i);
    end
    for i = 1:length(temp_EEG)
        EEG{end+1} = temp_EEG{i};
    end   
    clear temp_EEG
end

%% PREPROCESSING
%     In this section, choose one of the time segments to run by removing
%     the "%" symbol.
    %% 1-s segment
% for i = 1:length(EEG)
%     EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%     for j = 1:round(length(EEG_4s))/fs
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:j*fs)', fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:j*fs)', fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = (LABEL(1,i) + 1);
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% 1-second overlapping 0.5-second
% for i = 1:length(EEG)
%     EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%     for j = 1:7
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(EEG_4s(:,(j-1)*(fs-100)+1:j*fs-(j-1)*(fs-100))', fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(EEG_4s(:,(j-1)*(fs-100)+1:j*fs-(j-1)*(fs-100))', fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = (LABEL(1,i) + 1);
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% 2-second overlapping 1-second
% for i = 1:length(EEG)
%     EEG_4s = EEG{i}(:,201:1000);%only take 4-second MI
%     for j = 1:3
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:(j+1)*fs)', fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(EEG_4s(:,(j-1)*fs+1:(j+1)*fs)', fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = (LABEL(1,i) + 1);
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% Whole segment
for i = 1:length(EEG)
        %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%     data_4class{i} = eeg_filter(EEG{i}(:,:)', fs,[4 40])';
        %% Case 02: Apply feature selections: CSP
    data_4class{i} = eeg_filter(EEG{i}(:,:)', fs,[8 14])';
    
    data_4class{i} = data_4class{i}(:,201:1000);%only take 4-second MI
end
data_4class = data_4class'; 
label_4class = (LABEL + 1)';

fprintf('Preprocessing Done\n');

%% 4-CLASS OVA - KFOLD - FEATURE EXTRACTION - CLASSIFICATISON
%% 4-CLASS OVA - KFOLD
class=[1; 2; 3; 4];     %(1)RH  (2)LH  (3)RF  (4)LF
fprintf('OVA 4-class Classification (%d fold-cross validation)\n',FOLD);
CV= cvpartition(label_4class,'Kfold',FOLD);
cfm_LDA=[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];
acc=[];
for fold=1:CV.NumTestSets
    tic
    clear data label data_train data_test label_train label_test y_final
    data_train = data_4class(CV.training(fold)==1,:);  
    data_test = data_4class(CV.test(fold)==1,:); 
    label_train = label_4class(CV.training(fold)==1);
    label_test = label_4class(CV.test(fold)==1);

    for i=1:size(class,1)
        clear epoch_train epoch_test y_train x_train x_test score
        epoch_train = data_train;
        epoch_test = data_test;
        y_train = label_train;
        y_train(find(y_train~=class(i)))=5;
        y_train(find(y_train==class(i)))=1;
        y_train(find(y_train==5))=2;
        
        %% FEATURE EXTRACTION
%         In this block, choose a feature extraction method to process the
%         data. The way to choose is to remove the "%" symbol in that
%         method code.
            %% CSP
       trainParams.m = 3; % 2m <= channels
       [WCSP, x_train, x_test] = CSP_training(epoch_train, y_train, epoch_test, trainParams);
       band_selected(fold)={'None'};
       SaveFileName='CSP';

            %% FBCSP 
%        params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%             20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%        params.fs=fs;
%        params.m = 3;
%        [x_train, x_test]=FBCSP_training(epoch_train, y_train, epoch_test, params);
%        band_selected(fold)={'None'};
%        SaveFileName='FBCSP';

            %% DFBCSP Fisher
%        params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%                      20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%        params.fs=fs;
%        params.m = 3;
%        [x_train, x_test, bands_selected] = DFBCSP_training_Fisher(epoch_train, y_train, epoch_test, params);
%        band_selected(fold)= {bands_selected};
%        SaveFileName='DFBCSP_Fisher';

            %% DFBCSP mRmR
%        params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%               20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%        params.fs=fs;
%        params.m = 3;
%        [x_train, x_test, bands_selected] = DFBCSP_training_mRmR(epoch_train, y_train, epoch_test, params);
%        band_selected(fold)= {bands_selected};
%        SaveFileName='DFBCSP_mRmR';

       %% DFBCSP FmRmR
%        params.bands=[4 8; 6 10; 8 12; 10 14; 12 16; 14 18; 16 20; 18 22; ...
%               20 24; 22 26; 24 28; 26 30; 28 32; 30 34; 32 36; 34 38; 36 40];
%        params.fs=fs;
%        params.m = 3;
%        [x_train, x_test, bands_selected] = DFBCSP_training_FmRmR(epoch_train, y_train, epoch_test, params);
%        band_selected(fold)={bands_selected};
%        SaveFileName='DFBCSP_FmRmR';

       %% CLASSIFICATION
       switch i
           case 1
            model1 = fitcdiscr(x_train,y_train);
            [~,score1] = predict(model1, x_test);
           case 2
            model2 = fitcdiscr(x_train,y_train);
            [~,score2] = predict(model2, x_test);
           case 3
            model3 = fitcdiscr(x_train,y_train);
            [~,score3] = predict(model3, x_test);
           case 4
            model4 = fitcdiscr(x_train,y_train);
            [~,score4] = predict(model4, x_test);
       end
%        bands_pair(i) = {bands_selected};
%        WCSP_pair(i) = {WCSP};
    end

    score = [score1(:,1) score2(:,1) score3(:,1) score4(:,1)];
    %% Voting method
    for j = 1:size(score,1)
        [~,y_final(j,:)] = max(score(j,:));
    end

    A11 =  sum((y_final==1) & (label_test == 1));
    A12 =  sum((y_final==1) & (label_test == 2));
    A13 =  sum((y_final==1) & (label_test == 3));
    A14 =  sum((y_final==1) & (label_test == 4));
    A21 =  sum((y_final==2) & (label_test == 1));
    A22 =  sum((y_final==2) & (label_test == 2));
    A23 =  sum((y_final==2) & (label_test == 3));
    A24 =  sum((y_final==2) & (label_test == 4));
    A31 =  sum((y_final==3) & (label_test == 1));
    A32 =  sum((y_final==3) & (label_test == 2));
    A33 =  sum((y_final==3) & (label_test == 3));
    A34 =  sum((y_final==3) & (label_test == 4));
    A41 =  sum((y_final==4) & (label_test == 1));
    A42 =  sum((y_final==4) & (label_test == 2));
    A43 =  sum((y_final==4) & (label_test == 3));
    A44 =  sum((y_final==4) & (label_test == 4));
    cfm = [A11 A12 A13 A14; A21 A22 A23 A24; A31 A32 A33 A34; A41 A42 A43 A44]; 
    cfm_LDA = cfm_LDA + cfm; 

    cfm_result(fold)={cfm};
    BandSelected(fold)={band_selected};
    miniTime(fold)={toc}; 
    varName(fold)=cellstr(sprintf('FOLD %d',fold));

end
%     result.cfm{i}=cfm_LDA;
%accuracy
acc_cfm = 100*trace(cfm_LDA)/sum(cfm_LDA(:));
%F1-score
precision=cfm_LDA(1,1)/sum(cfm_LDA(1,:));
recall=cfm_LDA(1,1)/sum(cfm_LDA(:,1));
F1=(2 * precision * recall) / (precision + recall);
%display
fprintf('accuracy= %.1f; ',acc_cfm);
fprintf('F1-score= %.4f \n',F1);
disp(cfm_LDA);
%% Save result into workspace
    % (FileName(1:end-4)) = 'TR08_429_raw' => remove '.mat' -- FIXED
table=cell2table([miniTime; cfm_result; band_selected],...
'VariableNames',varName','RowNames',{'Time','cfm','Band selected'});
final_result.all.cfm_LDA=cfm_LDA;
final_result.all.acc=round(acc_cfm,2);
final_result.all.F1=round(F1,2);
final_result.all.Table=table;
%% ================================================================= %%
%% SAVE RESULT
%% Save File
eval(sprintf('%s=final_result',SaveFileName));
Files_check_semi = dir(fullfile([path],'result'));
if isempty(Files_check_semi)
    mkdir([path,'result']);
else
end
clear Files_check_semi
Files_check_final = dir(fullfile([path],'result',sprintf('final_result_%s.mat',FileName(1:4))));
if isempty(Files_check_final)
    save([path,'result','\',sprintf('final_result_%s.mat',FileName(1:4))],SaveFileName);
else
    save([path,'result','\',sprintf('final_result_%s.mat',FileName(1:4))],SaveFileName,'-append');
end