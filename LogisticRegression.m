function [J,Thetas]= LogisticRegression (X,y_norm,powers)
alpha=0.001;
Thetas=[0.5 0.5 0.5 0.5 0.5];
g= Thetas(1)*X(:,1).^powers + Thetas(2)*X(:,2).^powers + Thetas(3)*X(:,3).^powers + Thetas(4)*X(:,4).^powers + Thetas(5)*X(:,5).^powers;
Hypo=(1/(1+exp(-g)))';
h=1;
a=1;
J(h)=(1/(2*250))*sum((Hypo-y_norm).^2);

while a==1
    
    for i=1:1:5
       
        Thetas(i)=Thetas(i)-(alpha/250)*sum((Hypo-y_norm).*X(:,i));
         
    end
    %Update the hypothesis with the new thetas
        Hypo= Thetas(1)*X(:,1).^powers + Thetas(2)*X(:,2).^powers + Thetas(3)*X(:,3).^powers + Thetas(4)*X(:,4).^powers + Thetas(5)*X(:,5).^powers;
    %Calculate the cost function with the new thetas
        h=h+1;
        J(h)=(1/(2*250))*sum((Hypo-y_norm).^2);
        
        if J(h-1)-J(h)<10^-5
            a=2;
        end
end

end