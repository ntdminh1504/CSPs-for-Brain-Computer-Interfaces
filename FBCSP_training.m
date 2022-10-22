function [ftr, fts]=FBCSP_training(TRDATA,TRLB,TSDATA,params)
% Description:  Performs Filter Bank Common Spatial Pattern (FBCSP) on the
%               input data.
%               
% Author:       Minh Tran Duc Nguyen
%               
% Date:         October, 2022

bands=params.bands;
F=size(bands,1);
C=numel(unique(TRLB)); %number of classes
KTR=numel(TRDATA);
KTS=numel(TSDATA);

%% filter bank
for s=1:F
    for k=1:KTR
        x=TRDATA{k};
        Xtr_F{s}{k} = eeg_filter(x', params.fs, bands(s,1:2))' ; % channels x 3001
    end
    for k=1:KTS
        x=TSDATA{k};
        Xts_F{s}{k} = eeg_filter(x', params.fs, bands(s,1:2))' ; % channels x 3001
    end
%     fprintf('Filter bank: %d/%d \n', s,F); 
    
end


%% CSP feature
ftr=[];
for k=1:KTR
    f=[];
    for s=1:F
        W1{s} = train_csp(Xtr_F{s}, TRLB, params.m);
        X=Xtr_F{s}{k};
        OUT{k}=W1{s}*X; % N_channels x 3001 
        f=[f log(var(OUT{k},0,2)'/sum(var(OUT{k},0,2)))];
    end
    ftr = [ftr; f];
%     fprintf('[band %d/%d] extract CSP train_feature: %d/%d \n', s,F, k, KTR); 
end

fts=[];
for k=1:KTS
    f=[];
    for s=1:F
        X=Xts_F{s}{k};
        OUT{k}=W1{s}*X; % N_channels x 3001 
        f=[f log(var(OUT{k},0,2)'/sum(var(OUT{k},0,2)))];
    end
    fts = [fts; f];
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








