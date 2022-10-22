% Description:  This code is used to process dataset 2.
%               
% Author:       Minh Tran Duc Nguyen
%               
% Date:         October, 2022

%% ================================================================= %%
clear all;
clc;
warning off
close all;

%% CONFIG 
idname='[11]ManhDuyen';   %change subject ID
fs=500;
FOLD=10;

%% ================================================================= %%
%% IMPORTING DATA INTO MATLAB
selpath=uigetdir;
path_before =[selpath,'\',idname]; 
check_name=dir(path_before); 
check_name=check_name(3:length(check_name)); 
if strcmp(check_name(length(check_name)).name,sprintf('final_result_%s.mat',idname))
    check_name=check_name(1:end-1);
else
end
for len=1:length(check_name) 
    DAYS=check_name(len).name; 
    path =[selpath,'\',idname,'\',DAYS,'\']; 
    if ~isempty(dir([path])) 
        mkdir([path,'Files']); 
        mkdir([path,'trigger']);
        mkdir([path,'eye']);
        aaa= dir(fullfile([path],'*.csv')); 
        bbb= dir(fullfile([path],'*.txt'));
        for i=1:length(aaa)
            if isempty(strfind(aaa(i).name,'eye'))
                movefile([path,aaa(i).name],[path,'trigger']);
            else
                movefile([path,aaa(i).name],[path,'eye']);
            end
        end
        for i=1:length(bbb)
            if isempty(strfind(bbb(i).name,'result'))
                movefile([path,bbb(i).name],[path,'Files']);
            end
        end
    end
    disp('Done process folder');
    disp(path);

    %% TRIGGER + EVENT
    trigger_note = dir(fullfile([path,'trigger\'],'*.csv'));
    listfull = dir(fullfile([path,'Files\'],'*.txt'));

    list_event={}; list_file={}; %must be the same!
    for k=1:length(listfull)
        if ~isempty(strfind(listfull(k).name,'event'))
            list_event=[list_event ; listfull(k).name]; 
        else
            list_file=[list_file ; listfull(k).name]; 
        end
    end

    for k=1:length(list_file) % check delay trigger  
        fn = strtok(list_file{k},'.');
        fid=fopen([path,'Files\',list_event{k}]);
        txt = textscan(fid,'%s','delimiter','\n');
        delay_trigger(k,1) = str2num(txt{1,1}{23,1}(20))*60+...
            str2num(txt{1,1}{23,1}(22:23)) ;

        fclose('all');  
    end
    disp('-----------------------');
    disp('[1]***Loading All Files***');

    %% DATA
    sig_full=[]; 
    trigger_full=[];

    for k=1:length(list_file) 
            fn = strtok(list_file{k},'.');
            filename=fn ;  filename(strfind(filename,'_'))='-' ;
            temp_train= strfind(fn,'_');
            S=dlmread([path,'Files\',fn,'.txt'],'',1);  % read file and remove 1st line
            check = importdata([path,'trigger\',trigger_note(k).name]);
            if delay_trigger(k,1)==0
                sig_raw=S(1:end,:);
            else
                sig_raw=S(fs*delay_trigger(k,1):end,:); % data started
            end
            sig = sig_raw ;

            trigger_create = zeros(size(sig,1),1);
            for j=1:size(check,1) %All trials, 3 classes
                if size(check,2)==3
                    trigger_create(floor(check(j,2)*fs)) = check(j,1) ;
                else
                    if check(j,4)==13 %Mark noise
                        continue;
                    else
                        trigger_create(floor(check(j,2)*fs)) = check(j,1) ; 
                    end           
                end
            end

            temp_sig{k}= sig; 
            temp_trig{k}= trigger_create ;

            if fn(end)=='I'
                sig_full = [sig_full; sig]; 
                trigger_full = [trigger_full; trigger_create];
            elseif fn(end)=='A'
               sig_action = sig ;
               trigger_action = trigger_create ; 
            end

    disp(['load File : ',fn]);

    end
    eeg = sig_full(:,1:6); %C3-Cz-C4-P3-Pz-P4
    ecg = sig_full(:,end); %ECG
    %% SEGMENTATION
    disp('------------------------------------');
    disp('[2]***Segmentation (trials / trigger)***');

    event = [[find(trigger_full==1) ones(length(find(trigger_full==1)),1)] ;...
        [find(trigger_full==2) 2*ones(length(find(trigger_full==2)),1)] ; ...
        [find(trigger_full==3) 3*ones(length(find(trigger_full==3)),1)]] ;          % exactly cue

    %convert newdata for each type
    Ns = -3*fs ; 
    Nd = 9*fs  ;
    eeg_split =[]; 
    count =0; 
    check_event=[];
    event_choose=[];
    split=[];
    for j=1:size(event,1)
       if event(j,1)+Ns >=1 & event(j,1)+Nd <=size(eeg,1) & abs(eeg(event(j,1)+[Ns:Nd],:)) <=100 

           count = count +1;
           trials_eeg{count,1} = eeg(event(j,1)+[Ns:Nd],:) ;
           trials_ecg{count,1} = ecg(event(j,1)+[Ns:Nd],:) ;
           eeg_split =[eeg_split ; trials_eeg{count,1}];
           temp1= event(j,1)+[Ns:Nd]; 
           te{j,1}=temp1(1); te{j,2}=temp1(end);
           event_choose=[event_choose; event(j,1) event(j,2)];


           for k=1:3
               if event(j,2)==k
                   split{k}=size(eeg_split,1);
               end
           end

       end
    end

    for i=1:3
        count_each(i) = length(find(event_choose(:,2)==i));
    end

    info_trial=['Trials = ',num2str(count),' || class(1)= ',num2str(count_each(1)),...
        ' || class(2)= ',num2str(count_each(2)),' || class(3)= ',num2str(count_each(3))];
    disp(info_trial);

    %% PRE-PROCESSING
%     In this section, choose one of the time segments to run by removing
%     the "%" symbol.
        %% 1-s segment
%     for i = 1:length(trials_eeg)
%         trials_eeg{i} = trials_eeg{i}(3*fs:9*fs,1:6);
%         for j = 1:round(length(trials_eeg{i})/fs)
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:j*fs,1:6), fs, [4 40])';   % data (N_channels x Sample)
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:j*fs,1:6), fs, [8 14])';   % data (N_channels x Sample)
%             data_3class{i,j} = data_temp;
%             label_3class(i,j) = event_choose(i,2);
%         end
%     end
%     data_3class = reshape(data_3class,[size(data_3class,1)*size(data_3class,2),1]);
%     label_3class = reshape(label_3class,[size(label_3class,1)*size(label_3class,2),1]);

        %% 1-second overlapping 0.5-second
%     for i = 1:length(trials_eeg)
%         trials_eeg{i} = trials_eeg{i}(3*fs:9*fs,1:6);
%         for j = 1:11 % Divide the complete trial into 11 small trials of 1s0.5o according to dataset 2
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(trials_eeg{i}((j-1)*(fs-250)+1:j*fs-(j-1)*(fs-250),1:6), fs, [4 40])';
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(trials_eeg{i}((j-1)*(fs-250)+1:j*fs-(j-1)*(fs-250),1:6), fs, [8 14])';
%             data_3class{i,j} = data_temp;
%             label_3class(i,j) = event_choose(i,2);
%         end
%     end
%     data_3class = reshape(data_3class,[size(data_3class,1)*size(data_3class,2),1]);
%     label_3class = reshape(label_3class,[size(label_3class,1)*size(label_3class,2),1]);

        %% 2-second overlapping 1-second
%     for i = 1:length(trials_eeg)
%         trials_eeg{i} = trials_eeg{i}(3*fs:9*fs,1:6);
%         for j = 1:5 % Divide the complete trial into 5 small trials of 2s1o according to dataset 2
%                 %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%             data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:(j+1)*fs,1:6), fs, [4 40])';
%                 %% Case 02: Apply feature selections: CSP
% %             data_temp = eeg_filter(trials_eeg{i}((j-1)*fs+1:(j+1)*fs,1:6), fs, [4 40])';
%             data_3class{i,j} = data_temp;
%             label_3class(i,j) = event_choose(i,2);
%         end
%     end
%     data_3class = reshape(data_3class,[size(data_3class,1)*size(data_3class,2),1]);
%     label_3class = reshape(label_3class,[size(label_3class,1)*size(label_3class,2),1]);

        %% Whole segment
    for i=1:length(trials_eeg)
            %% Case 01: Apply feature selections: FBCSP, DFBCSP Fisher, DFBCSP mRmR, DFBCSP FmRmR
%         data_3class{i,1} = eeg_filter(trials_eeg{i}(3*fs:9*fs,1:6), fs, [4 40])';   % data (N_channels x Sample)
            %% Case 02: Apply feature selections: CSP
        data_3class{i,1} = eeg_filter(trials_eeg{i}(3*fs:9*fs,1:6), fs, [8 14])';   % data (N_channels x Sample)
        
        label_3class(i,1) =  event_choose(i,2);
    end
    
    fprintf('Preprocessing Done\n');

%% 3-CLASS OVA - KFOLD - FEATURE EXTRACTION - CLASSIFICATISON
    %% 3-CLASS OVA - KFOLD
class=[1; 2; 3]; %RH(1) - LH(2) - F(3)
fprintf('OVA 3-class Classification (%d fold-cross validation)\n',FOLD);
CV= cvpartition(label_3class,'Kfold',FOLD);
cfm_LDA=[0 0 0; 0 0 0; 0 0 0];
acc=[];
for fold=1:CV.NumTestSets
    fprintf('FOLD: %d \n',fold);
    tic
    clear data label data_train data_test label_train label_test y_predicted y_final
    data_train = data_3class(CV.training(fold)==1,:);  
    data_test = data_3class(CV.test(fold)==1,:); 
    label_train = label_3class(CV.training(fold)==1);
    label_test = label_3class(CV.test(fold)==1);
    
    %% FEATURE EXTRACTION
%     In this block, choose a feature extraction method to process the
%     data. The way to choose is to remove the "%" symbol in that method
%     code.
    for i=1:size(class,1)
        
        epoch_train = data_train;
        epoch_test = data_test;
        y_train = label_train;
        y_train(find(y_train~=class(i)))=5;
        y_train(find(y_train==class(i)))=1;
        y_train(find(y_train==5))=2;

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
%        band_selected(fold) = {bands_selected};
%        SaveFileName='DFBCSP_FmRmR';
       
       %% Classification
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
       end
    end
    score = [score1(:,1) score2(:,1) score3(:,1)];
    %%Voting method
    for j = 1:size(score, 1)
        [~,y_final(j,:)] = max(score(j,:));
    end
    
    A11 =  sum((y_final==1) & (label_test == 1));
    A12 =  sum((y_final==1) & (label_test == 2));
    A13 =  sum((y_final==1) & (label_test == 3));
    A21 =  sum((y_final==2) & (label_test == 1));
    A22 =  sum((y_final==2) & (label_test == 2));
    A23 =  sum((y_final==2) & (label_test == 3));
    A31 =  sum((y_final==3) & (label_test == 1));
    A32 =  sum((y_final==3) & (label_test == 2));
    A33 =  sum((y_final==3) & (label_test == 3));
    cfm = [A11 A12 A13; A21 A22 A23; A31 A32 A33]; 
    cfm_LDA = cfm_LDA + cfm; 

    cfm_result(fold)={cfm}; 
    BandSelected(fold)={band_selected};
    miniTime(fold)={toc}; 
    varName(fold)=cellstr(sprintf('FOLD %d',fold));

end
result.cfm=cfm_LDA;
%Accuracy
acc_cfm = 100*trace(cfm_LDA)/sum(cfm_LDA(:));
%F1-score
precision=cfm_LDA(1,1)/sum(cfm_LDA(1,:));
recall=cfm_LDA(1,1)/sum(cfm_LDA(:,1));
F1=(2 * precision * recall) / (precision + recall);
%Display
fprintf('accuracy= %.1f; ',acc_cfm);
fprintf('F1-score= %.2f \n',F1);
disp(cfm_LDA);

%save result into workspace
if isvarname(DAYS)==0
  temp_name=strsplit(DAYS,'_');
  days=char(strcat('month',temp_name(1),'day',temp_name(2)));
else
  days=DAYS;
end
  table=cell2table([miniTime; cfm_result; band_selected],...
  'VariableNames',varName','RowNames',{'Time','cfm','Band selected'});
  final_result.(days).cfm_LDA=cfm_LDA;
  final_result.(days).acc=round(acc_cfm,2);
  final_result.(days).F1=round(F1,2);
  final_result.(days).Table=table;
  clearvars -except idname fs FOLD selpath check_name final_result SaveFileName
end

eval(sprintf('%s=final_result',SaveFileName));
%% ================================================================= %%
%% SAVE RESULT
%% Auto create save folder
temp_selpath=pwd;
check = dir(fullfile(temp_selpath, 'Dataset2_final_result'));
if isempty(check)
    mkdir([temp_selpath,'\','Dataset2_final_result']);
else
end
temp_selpath=[temp_selpath,'\','Dataset2_final_result','\'];
clear check
%% Auto create folder contain subject
check = dir(fullfile([temp_selpath,'\',idname]));
if isempty(check)
    mkdir([temp_selpath,'\',idname]);
else
end
selpath=[temp_selpath,'\',idname,'\'];
clear check temp_selpath
%% Save
Files_check_final = dir(fullfile([selpath,'\'],sprintf('final_result_%s.mat',idname)));
if isempty(Files_check_final)
    save([selpath,'\',sprintf('final_result_%s.mat',idname)],SaveFileName);
else
    save([selpath,'\',sprintf('final_result_%s.mat',idname)],SaveFileName,'-append');
end
