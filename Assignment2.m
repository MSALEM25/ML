data = tabularTextDatastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
featuers=read(data);

x1=table2array(featuers(:,1));
x2=table2array(featuers(:,2));
x3=table2array(featuers(:,3));
x4=table2array(featuers(:,4));
x0=ones(250,1);

x1_norm=x1/max(x1);
x2_norm=x2/max(x2);
x3_norm=x3/max(x3);
x4_norm=x4/max(x4);

X=[x0 x1_norm x2_norm x3_norm x4_norm];
%
y=table2array(featuers(:,14));
y_norm=y/max(y);

[J,Thetas]=LogisticRegression(X,y_norm,4);
%2nd hypo.
[J1,Thetas1]=LogisticRegression(X,y_norm,5);
%3rd hypo.
[J2,Thetas2]=LogisticRegression(X,y_norm,7);
%4th hypo.
[J3,Thetas3]=LogisticRegression(X,y_norm,4);
figure(1)
plot(J);
figure(2)
plot(J1);
figure(3)
plot(J2);
figure(4)
plot(J3);
