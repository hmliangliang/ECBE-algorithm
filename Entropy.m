function e = Entropy( target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%该函数计算数据集的熵值
 s=0;%记录熵值
 k=max(target);%找出最大的类标签
 num=zeros(1,k);%num记录数据集中每个类的统计值
for i=1:size(target,1)
    num(1,target(i,1))=num(1,target(i,1))+1;
end
for j=1:size(num,2)%计算熵值
    if num(1,j)==0
        num(1,j)=size(target,1);
    else
        s=s-(num(1,j)/(size(target,1)))*log2(num(1,j)/(size(target,1)));
    end
end
e=s;
end

