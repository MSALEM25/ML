clear all
clear
clc
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',18000);
T = read(ds);
TrainingSet=T(1:1080,:);
size(T);
Alpha=.01;

M=length(T{:,1});

U0train=T{1:1080,2};
U0train=(U0train-mean(U0train))./std(U0train);

U0CV=T{1081:1500,2};
U0CV=(U0CV-mean(U0CV))./std(U0CV); %Normalizing output


UTrain=T{1:1080,6:10};
UCV=T{1081:1800,6:10};

LenTrain = length(UTrain);
LenCV = length(UCV);
z=length(UTrain);
XTrain=[ones(z,1) UTrain UTrain.^2];
XCV=[ones(length(UCV),1) UCV UCV.^2];

nTrain=length(XTrain(1,:));
for W=2:nTrain
    if max(abs(XTrain(:,W)))~=0
    XTrain(:,W)=(XTrain(:,W)-mean((XTrain(:,W))))./std(XTrain(:,W));
    end
end
nCV=length(XCV(1,:));
for W=2:nCV
    if max(abs(XCV(:,W)))~=0
    XCV(:,W)=(XCV(:,W)-mean((XCV(:,W))))./std(XCV(:,W));
    end
end

YTrain=TrainingSet{:,3}/mean(TrainingSet{:,3});
YCV=TrainingSet{11,3}/mean(TrainingSet{11,3});

ThetaTrain=zeros(nTrain,1);
ThetaCV=zeros(nCV,1);
% for =1:500
% 
% ETrain(k)=(1/(2*m))*sum((XTrain*ThetaTrain-Y).^2);
% ECV(k)=(1/(2*m))*sum((XCV*ThetaCV-Y).^2);
% end
X=1;
Y=1;
ETrain=[];
ECV=[];
while Y==1
Alpha=Alpha*1;
ThetaTrain=ThetaTrain-(Alpha/M)*XTrain'*(XTrain*ThetaTrain-YTrain);
ThetaCV=ThetaCV-(Alpha/M)*XCV'*(XCV*ThetaCV-YCV);

ETrain=(1/(2*M))*sum((XTrain*ThetaTrain-YTrain).^2);
ETrain=[ETrain;ETrain];
ECV(X)=(1/(2*M))*sum((XCV*ThetaCV-YCV).^2);
ECV=[ECV;ECV];
 X=X+1
if ETrain(X-1)-ETrain(X)<0
   
    break
end
S=(ETrain(X-1)-ETrain(X))./ETrain(X-1);
if S <.000001
    Y=0;
end
end

figure (1)
plot(X,ECV,'black')
hold on
plot(X,ETrain,'red')
legend('CV','Train')
title('House Price')
ylabel('Cost Fun')
xlabel('Iter')

