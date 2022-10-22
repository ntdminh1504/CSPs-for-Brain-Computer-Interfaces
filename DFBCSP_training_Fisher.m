function [ftr, fts, bands_selected_1, W1] = DFBCSP_training_Fisher(TRDATA,TRLB,TSDATA,params)
% Description:  Performs DFBCSP - Fisher's ratio on the input data.
%               
% Author:       Minh Tran Duc Nguyen
%               
% Date:         October, 2022

bands=params.bands;
F=size(bands,1);
C=numel(unique(TRLB)); %number of classes
KTR=numel(TRDATA);
KTS=numel(TSDATA);


%% (1) Train data using all channels --> spectral power of each subband (of each trial)
for s=1:F
    for k=1:KTR
       x=TRDATA{k}; %channels x samples
       Xtr{s}{k} = eeg_filter(x', params.fs, bands(s,1:2)); % samples x channels
       Parent_FB{s,k} = (1/length(Xtr{s}{k}))*sum(Xtr{s}{k}.^2); %{bands x trials}(channel column)
       Xtr{s}{k} = Xtr{s}{k}';
    end
end

%% (2) calculate Fisher Ratio of each subband
Fisher_ratio = zeros(F, size(x,1));
for s=1:F
    SWc=zeros(C,size(x,1));
    SBc=zeros(C,size(x,1));
    temp=[];
    for c=1:C
        idx = find(TRLB==c);
        P = [];
         for i=1:length(idx)
            P = [P; Parent_FB{s,idx(i)}];
         end
         for n=1:length(P)
            SWc(c,:) = SWc(c,:) + (P(n,:)-mean(P)).^2;
         end
         temp = [temp; mean(P)]; %accumulation of mean(Parent_FB)
    end
    for c=1:C
        idx = find(TRLB==c);
        SBc(c,:) = length(idx)*((mean(temp)-temp(c,:)).^2);
    end
    SW = sum(SWc); %within-class variance
    SB = sum(SBc); %between-class variance
    Fisher_ratio(s,:) = SB./SW;
end
%Fisher_ratio (m filterbanks x n channels)
[~,z]=sort(Fisher_ratio,'descend');
bands_selected = bands(z(1,:));
bands_selected_1 = [unique(bands_selected); unique(bands_selected+4)]';
F_selected_1 = size(bands_selected_1,1);

%% filter bank of bands_selected
for s=1:F_selected_1
    for k=1:KTR
        Xtr_F{s}{k} = Xtr{(find(bands(:,1) == bands_selected_1(s,1)))}{k}; % channels x samples
    end
    for k=1:KTS
        x=TSDATA{k};
        Xts_F{s}{k} = eeg_filter(x', params.fs, bands_selected_1(s,1:2))'; % channels x samples
    end
%     fprintf('\nFilter band selected based Fisher: %d-%d \n', s,F_selected);
 
end
%% CSP extraction
%ftr = ftr(:);
ftr=[];
for k=1:KTR
    f=[];
    for s=1:F_selected_1
        W1{s} = train_csp(Xtr_F{s}, TRLB, params.m);
        X=Xtr_F{s}{k};
        OUT{k}=W1{s}*X; % N_channels x samples 
        f=[f log(var(OUT{k},0,2)'/sum(var(OUT{k},0,2)))];
    end
    ftr = [ftr; f];
%     fprintf('[band %d/%d] extract CSP train_feature: %d/%d \n', s,F, k, KTR); 
end

fts=[];
for k=1:KTS
    tic
    f=[];
    for s=1:F_selected_1
        X=Xts_F{s}{k};
        OUT{k}=W1{s}*X; % N_channels x 3001 
        f=[f log(var(OUT{k},0,2)'/sum(var(OUT{k},0,2)))];
    end
    fts = [fts; f];
    testtime = toc; %only 1 test trial
%     fprintf('[band %d/%d] extract CSP test_feature: %d/%d \n', s,F, k, KTS); 
end

end


%-----------------------train_csp-----------------------------%
function [WCSP]=train_csp(TRDATA,TRLB,m)
    %number of classes
    C=numel(unique(TRLB));
    %number of channels
    N=size(TRDATA{1},1);
    %number of epochs in training set
    K=numel(TRDATA);
    %number of epochs in test set
    % KTS=numel(TSDATA);
    RA=zeros(N,N);
    RB=zeros(N,N);
    na=0;
    nb=0;
    for c=1:C
        R{c}=zeros(N,N);
        n(c)=0;
    end
    %% calculate average covariance matrices
    for k=1:K
        X=TRDATA{k};
        r=X*X'/trace(X*X');
        c=TRLB(k);
        R{c}=R{c}+r;
        n(c)=n(c)+1;
    end
    for c=1:C
        R{c}=R{c}/n(c);
    end
    %% calculate spatial filters for each class
    WCSP=[]; % csp matrix
    for c=1:C
        %calculate num & den
        RA=R{c};
        RB=zeros(N,N);
        for ci=1:C
            if ci~=c
                RB=RB+R{ci};
            end
        end
        %calculate CSP matrix
        Q=inv(RB)*RA;
        [W A]=eig(Q);
        %sort eigenvalues in descending order
        [A order] = sort(diag(A),'descend');
        % sort eigen vectors
        % W=inv(W)';
        W = W(:,order);
        WCSP=[WCSP;W(:,1:m)'];
        L{c}=A(1:m);
    end
end