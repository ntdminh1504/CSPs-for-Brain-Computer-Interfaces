function [WCSP, ftr, fts]=CSP_training(TRDATA,TRLB,TSDATA,trainParams)
% Description:  Performs common spatial pattern (CSP) on the input data.
%               
% Author:       Minh Tran Duc Nguyen
%               
% Date:         October, 2022

%number of classes
C=numel(unique(TRLB));
%number of channels
N=size(TRDATA{1},1);
%number of epochs in training set
K=numel(TRDATA);
%number of epochs in test set
% KTS=numel(TSDATA);

% RA=zeros(N,N);
% RB=zeros(N,N);
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
    [W1 A1]=eig(Q);

    %sort eigenvalues in descending order
    [A order] = sort(diag(A1),'descend');
    % sort eigen vectors
    % W=inv(W)';
    W = W1(:,order);
    WCSP=[WCSP;W(:,1:trainParams.m)'];
    L{c}=A(1:trainParams.m);
    
end

%% feature extraction

% RR = R{1} + R{2} ;
% [a,b]= eig(RR); % RR=a*b*a'
% WW = sqrt(inv(b))*a' ; % whitening transformation 
% K1 = WW*R{1}*WW' ;  
% K2 = WW*R{2}*WW' ;
% [U1,lamda1] = eig(K1) ; [U2,lamda2] = eig(K2) ;
% P = (U1'*WW) ;

Ktr=numel(TRDATA);
for k=1:Ktr
    X=TRDATA{k};
    
%     ZTR{k}=P*X;
%     ftr(k,:)=[log(var(ZTR{k},0,2)'/sum(var(ZTR{k},0,2)))];
    
    ZTR{k}=WCSP*X;
    ftr(k,:)=[log(var(ZTR{k},0,2)'/sum(var(ZTR{k},0,2)))]; %feature train
    
end


Kts=numel(TSDATA);
for k=1:Kts
    X=TSDATA{k};
    
%     ZTS{k}=P*X;
%     fts(k,:)=[log(var(ZTS{k},0,2)'/sum(var(ZTS{k},0,2)))];
    
    ZTS{k}=WCSP*X;
    fts(k,:)=[log(var(ZTS{k},0,2)'/sum(var(ZTS{k},0,2)))];% feature test

end


%% Plot CSP variance
% d=[];
% for i=1:K
%     temp1 = [var(TRDATA{i},0,2)/sum(var(TRDATA{i},0,2))]; 
%     temp2 = [var(ZTR{i},0,2)'/sum(var(ZTR{i},0,2))];
%     d=[d ;temp1(1) temp1(2) temp2(1) temp2(2)];
% end
% id= find(TRLB==2); id=id(1);
% 
% figure(5);clf;
% subplot(221); hold on
% plot(d(1:id-1,1),d(1:id-1,2),'r*');
% plot(d(id:K,1),d(id:K,2),'b*');
% v=axis; axis([0 0.6 0 0.6]);
% 
% subplot(223); hold on
% plot(d(1:id-1,3),d(1:id-1,4),'r*');
% plot(d(id:K,3),d(id:K,4),'b*');
% v=axis; axis([0 0.6 0 0.6]);


%% CSP new
% RR = R{1} + R{2} ;
% [a,b]= eig(RR); % R=a*b*a'
% WW = sqrt(inv(b))*inv(a) ; % whitening transformation 
% K1 = WW*R{1}*WW' ;  
% K2 = WW*R{2}*WW' ;
% [U1,lamda1] = eig(K1) ; [U2,lamda2] = eig(K2) ;
% 
% P = (U1'*WW)' ;





