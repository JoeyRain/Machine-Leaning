N=132;
%initialize the data set
dataSet=zeros(132,2);
for i=1:N
    dataSet(i,1)=fix(i/12)*0.1+0.5;
    dataSet(i,2)=mod(i,12)*0.2+0.8;
    if(mod(i,12)==0)
        dataSet(i,1)=dataSet(i-1,1);
        dataSet(i,2)=3.2;
    end
end

k=11;   %specify the number of clusters
centers=cell(1,k);
for i=1:k
    %centers{1,i}=[rand,rand]*4; %specify the center of clusters by random
    centers{1,i}=[i+5,i+10]*0.1;
end

sections=ones(1,N);
flag=1;
centersTemp=cell(1,k);
while(flag)
    %divide the data set into corresponding sections by Euclide distance
    for i=1:N
        distance=sqrt((dataSet(i,1)-centers{1,sections(i)}(1,1)).^2+(dataSet(i,2)-centers{1,sections(i)}(1,2)).^2);
        for j=1:k
        distanceTemp=sqrt((dataSet(i,1)-centers{1,j}(1,1)).^2+(dataSet(i,2)-centers{1,j}(1,2)).^2);
        if(distanceTemp<distance)
            sections(i)=j;
        end
        end
    end

    %set the center of each clusters as the means of them
    for i=1:k
        sum=zeros(1,2);
        number=0;
        for j=1:N
            if(sections(j)==i)
                sum=sum+dataSet(j,:);
                number=number+1;
            end
        end
        %assert(number==0,"出现聚类无元素");
        if(number~=0)
            means=sum./number;
            centers{1,i}=means;
        end      
    end
    if(isequal(centers,centersTemp))
        break;
    else
        centersTemp=centers;
    end
end