function e = Entropy( target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%�ú����������ݼ�����ֵ
 s=0;%��¼��ֵ
 k=max(target);%�ҳ��������ǩ
 num=zeros(1,k);%num��¼���ݼ���ÿ�����ͳ��ֵ
for i=1:size(target,1)
    num(1,target(i,1))=num(1,target(i,1))+1;
end
for j=1:size(num,2)%������ֵ
    if num(1,j)==0
        num(1,j)=size(target,1);
    else
        s=s-(num(1,j)/(size(target,1)))*log2(num(1,j)/(size(target,1)));
    end
end
e=s;
end

