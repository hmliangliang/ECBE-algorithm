function m =MSE( data,target )
   %UNTITLED2 Summary of this function goes here
   %   Detailed explanation goes here
   %���㷨��MSE����5�۽�����֤�����м���
       k=size(data,1)/5;%ÿһ�����ݿ��ʵ������
       temp=[];
       for i=1:k:size(data,1)
            A=data;
            B=target;
            test=A(i:(i+k-1),:);
            testtarget=B(i:(i+k-1),:);
            A(i:(i+k-1),:)=[];
            B(i:(i+k-1),:)=[];
            res=C4_5(A',B',test',35,10);
            res=res';
            count=0;
            for j=1:size(res)
               if res(j,1)~=testtarget(j,1)
                  count=count+1;
               end
            end
           err=count/(size(res,1));
           temp=[temp,err];
       end
       m=sum(temp)/(size(temp,2));
 end