% Description:  This code is used to process dataset 1.
%               
% Author:       Minh Tran Duc Nguyen, Nhi Yen Xuan Phan
%               
% Date:         October, 2022
%% ================================================================= %%
clear all;
clc;
warning off
close all;
%% ================================================================= %%
%% CONFIG
FOLD=10;
%% IMPORTING DATA INTO MATLAB
[FileName,Path]=uigetfile('*.mat');
RawData=load([Path,FileName]);
temp=split(FileName,'.');
DataName=char(temp(1));
fprintf('----------------Step 1: Loading raw data----------------\n');
fprintf('Load %s data \n',DataName);
%% DATA
    % Time for 1 trial
t0=0; %a fixation cross appeared on the black screen, a short acoustic warning tone was presented
t1=2; t2=3.25; % a cue in the form of an arrow pointing
t3=3; t4=6; %motor imagery task until the fixation cross disappeared from the screen
t5=7; %A short break followed where the screen was black again
 
    % Segmentation
clear temp;
RawData=RawData.data(1,4:end); %take N run with N=length of rawData (different for subject A04!!!)
for i=1:length(RawData) %length(RawData)= the number of RUN
    k=1;
    for j=1:length(RawData{1,i}.trial)
        if ~RawData{1,i}.artifacts(j)
            if j ~= length(RawData{1,i}.trial)
                temp.rawdata{1,k}=RawData{1,i}.X(RawData{1,i}.trial(j):RawData{1,i}.trial(j+1)-1,:);
                temp.class(k)=RawData{1,i}.y(j);
                k=k+1;
            else
                temp.rawdata{1,k}=RawData{1,i}.X(RawData{1,i}.trial(j):end,:);
                temp.class(k)=RawData{1,i}.y(j);
            end
        end
    end
    if i==1
        data.rawdata=temp.rawdata;
        data.class=temp.class;
    else
        for j=1:length(temp.rawdata)
            data.rawdata{1,end+1}=temp.rawdata{1,j};
        end
        data.class(1,end+1:end+length(temp.class))=temp.class;
    end
    clear temp
end

data.label={'EEG-Fz';'EEG'...
        ;'EEG';'EEG';'EEG';'EEG';'EEG';'EEG-C3'...
        ;'EEG';'EEG-Cz';'EEG';'EEG-C4';'EEG';'EEG';'EEG'...
        ;'EEG';'EEG';'EEG';'EEG';'EEG-Pz';'EEG';'EEG'...
        ;'EOG-left';'EOG-central';'EOG-right'};
fs=RawData{1,1}.fs;
for i=1:4 %4 class
    count_each(i) = length(find(data.class==i));
end
clear temp;

%% PRE-PROCESSING
%     In this section, choose one of the time segments to run by removing
%     the "%" symbol.
    %% 1-s segment
% for i = 1:length(data.rawdata)
%     trials_eeg{i} = data.rawdata{1,i}(t3*fs:t4*fs,1:end-3);
%     for j = 1:round(length(trials_eeg{i})/fs)
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:j*fs,:), fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:j*fs,:), fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = data.class(i)';
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% 1-second overlapping 0.5-second
% for i = 1:length(data.rawdata)
%     trials_eeg{i} = data.rawdata{1,i}(t3*fs:t4*fs,1:end-3);
%     for j = 1:5 
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(trials_eeg{i}((j-1)*(fs-125)+1:j*fs-(j-1)*(fs-125),:), fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(trials_eeg{i}((j-1)*(fs-125)+1:j*fs-(j-1)*(fs-125),:), fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = data.class(i)';
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% 2-second overlapping 1-second
% for i = 1:length(data.rawdata)
%     trials_eeg{i} = data.rawdata{1,i}(t3*fs:t4*fs,1:end-3);
%     for j = 1:2
%         %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:(j+1)*fs,:), fs, [4 40])';
%         %% Case 02: Apply feature selections: CSP
% %         data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:(j+1)*fs,:), fs, [8 14])';
%         
%         data_4class{i,j} = data_temp;
%         label_4class(i,j) = data.class(i)';
%     end
% end
% data_4class = reshape(data_4class,[size(data_4class,1)*size(data_4class,2),1]);
% label_4class = reshape(label_4class,[size(label_4class,1)*size(label_4class,2),1]);

    %% Whole segment
for i=1:length(data.rawdata)
    %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%     data_4class{i,1} = eeg_filter(data.rawdata{1,i}(t3*fs:t4*fs,1:end-3), fs, [4 40]);   % data (N_channels x Sample)
    %% Case 02: Apply feature selections: CSP
    data_4class{i,1} = eeg_filter(data.rawdata{1,i}(t3*fs:t4*fs,1:end-3), fs, [8 14]);   % data (N_channels x Sample)
    
    data_4class{i,1} = data_4class{i,1}';
    label_4class(i,1) = data.class(i)';
end

fprintf('----------------Step 2: Processing raw data----------------\n');
fprintf('Done \n');

%% 4-CLASS OVA - KFOLD - FEATURE EXTRACTION - CLASSIFICATISON
%% 4-CLASS OVA - KFOLD
class=[1; 2; 3; 4];     %(1)LH  (2)RH  (3)BF  (4)T
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
result.cfm{i}=cfm_LDA;
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
    
%Create the result variable 'final_result'
table=cell2table([miniTime; cfm_result; band_selected],...
        'VariableNames',varName','RowNames',{'Time','cfm','Band selected'});
final_result.cfm_LDA=cfm_LDA;
final_result.acc=round(acc_cfm,2);
final_result.F1=round(F1,2);
final_result.Table=table;

%% ================================================================= %%
%% SAVE RESULT
eval(sprintf('%s=final_result',SaveFileName));
Files_check_final = dir(fullfile([Path], sprintf('final_result_%s.mat',FileName)));
if isempty(Files_check_final)
    save([Path,'\',sprintf('final_result_%s.mat',FileName)],SaveFileName);
else
    save([Path,'\',sprintf('final_result_%s.mat',FileName)],SaveFileName,'-append');
end
clearvars -except idname fs FOLD selpath check_name final_result SaveFileName FileName