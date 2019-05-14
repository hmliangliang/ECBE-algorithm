cu=SEA;%data set is SEA
col=size(cu,2);%���ݼ���ά��
data=cu(:,1:(col-1));
target=cu(:,col);
klabel=max(unique(target));%���ݼ������ǩ��Ŀ
winsize=100;
a=0.05;%������ˮƽ
b=2;%�ӷ���������Ȩֵ���������ӣ�b>1
r=0.1;%��������Ư�ƺ��¼����������Ȩֵ���������
j=0;%��¼��ǰ�������ĸ���
k=5;%ϵͳ�涨�ӷ������ķ�ֵ
s=[];%��¼ÿ���ӷ������Ե�ǰ���ݼ��ķ���������ֵ
theta=2*sqrt((log2(1/a))/(2000*100));
theta_weight=0;
tt=0;%��¼��ǰ���ԵĴ���
acc=[];%��¼ÿһ�β��Ե�׼ȷ��
detaH1=0;
detaH2=detaH1;
ensemble=struct('traindata',[],'traintarget',[],'weight',[]);
tic;
for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %�γ�ѵ����
        traindata=data((i-2*winsize+1):(i-winsize),:);
        traintarget=target((i-2*winsize+1):(i-winsize),:);
        if j<k %ϵͳ�еķ���������
            if j==0 %ϵͳ���޷�����
                ensemble(1).traindata=traindata;
                ensemble(1).traintarget=traintarget;
                ensemble(1).weight=1;
                j=j+1;
            else %ϵͳ���з�������δ�ﵽ����
                temp=struct('traindata',[],'traintarget',[],'weight',[]);
                temp.traindata=traindata;
                temp.traintarget=traintarget;
                temp.weight=1;
                ensemble=[ensemble,temp];
                j=j+1;
            end
        end
      if j~=0
            H1=Entropy(traintarget);%���㵱ǰ���ݿ����ֵ
            result=[];%��¼ÿ����������ѵ�����ķ�����
            for ei=1:size(ensemble,2)%�õ�ǰ��������traindata���з���
                model=ClassificationTree.fit(ensemble(ei).traindata,ensemble(ei).traintarget);
                res=predict(model,traindata);
                h=Entropy(res);%�ӷ������ķ���������ֵ
                s=[s,h];%����ÿһ���ӷ���������������ֵ
                ch=abs(h-H1);%�صı仯���
                %�����ӷ�������Ȩֵ
                if ch>0 %�з������������˥��Ȩֵ
                   ensemble(ei).weight=(1/(exp(1+ch)))*(ensemble(ei).weight);
                else %��ȫ������ȷ������Ȩֵ��b>1
                   ensemble(ei).weight=b*(ensemble(ei).weight);
                end
                result=[result,res];%�д���ʵ�����д��������
            end
            %���ݷ�������ȨͶƱ��������
            wr=zeros(size(traindata,1),klabel);%��¼ͶƱ��Ȩֵ״��
            for ex=1:size(result,1)%ͳ��ͶƱ���
                for ey=1:size(result,2)
                    wr(ex,result(ex,ey))=wr(ex,result(ex,ey))+ensemble(ey).weight;
                end
            end
            [c,last]=max(wr,[],2);%���ݼ�Ȩ����������ߣ�last��¼���߽��
            H2=Entropy(last);%H2������������ݼ�����ֵ
            detaH2=abs(H2-H1);
            if tt==0
                detaH1=detaH2;
            end
            %�ж��Ƿ�������Ư��
            if (abs(detaH1-detaH2)>theta)||(detaH1>theta)||(detaH2>theta) %�����˸���Ư��
                disp('����Ư��');
                [c,d]=min([ensemble.weight]);%ѡ��Ȩֵ��С�ķ�����
                ensemble(d)=[];%ɾ��
                j=j-1;
                if size(s,2)>1
                    u=mean(s);%���ӷ�������ֵ�ľ�ֵu
                    v=std(s);%���ӷ�������ֵ�ı�׼��v
                    theta_weight=u-(v/(sqrt(size(s,2)))*tinv(0.95,(size(s,2)-1)))-3*v;
                    s=[];
                else
                    theta_weight=0;
                end
                ef=1;
                while ef<=size(ensemble,2)%�Ե�ǰ������ϵͳ֮�У�Ȩֵ����theta_weight���ӷ�������Ҫɾ��
                    if ensemble(ef).weight<theta_weight
                       ensemble(ef)=[];%ɾ��������
                       j=j-1;
                    else
                       ef=ef+1;%��ǰ����������Ҫ�󣬼�����ǰɨ��
                    end
                end
                %�����·�����
                temp1=struct('traindata',[],'traintarget',[],'weight',[]);
                temp1.weight=r*sum([ensemble.weight]);
                if j<k %������δ�ﵽ����
                    ensemble=[ensemble,temp1];%����ѵ���ķ��������뵽ϵͳ��ȥ
                else
                    [c,d]=min([ensemble.weight]);%ѡ��Ȩֵ��С�ķ�����
                    ensemble(d)=[];%ɾ��
                    ensemble=[ensemble,temp1];
                end
            end
            detaH1=detaH2;
            for qa=1:size(ensemble,2)%��ѵ����ȥѵ��ÿһ��������
                ensemble(qa).traindata=[ensemble(qa).traindata;traindata];
                ensemble(qa).traintarget=[ensemble(qa).traintarget;traintarget];
            end
            %��ò��Լ�
            tt=tt+1;
            testdata=data((i-winsize+1):i,:);
            testtarget=target((i-winsize+1):i,:);
            cresult=[];%��¼ÿ���ӷ������Բ��Լ��ķ������
            for di=1:size(ensemble,2)
                cmodel=ClassificationTree.fit(ensemble(di).traindata,ensemble(di).traintarget);%��÷���ģ��
                cres=predict(cmodel,testdata);
                cresult=[cresult,cres];%�д���ʵ�����д��������
            end
            cw=zeros(size(testdata,1),klabel);%cw��¼����ʵ����ÿ�����ǩ�ϵ�ͶƱȨֵ
            for x=1:size(cresult,1)
                for y=1:size(cresult,2)
                    cw(x,cresult(x,y))=cw(x,cresult(x,y))+ensemble(y).weight;
                end
            end
            [c,clabel]=max(cw,[],2);%����Ȩ�����ԭ����������
            %ͳ����ȷ����ʵ������
            count=0;
            for g=1:size(clabel,1)
                if clabel(g,1)==testtarget(g,1)
                    count=count+1;
                end
            end
            right=count/(size(clabel,1));%׼ȷ��
            acc=[acc,right];
            disp(['��',num2str(tt),'�β��Ե���ȷ��Ϊ��',num2str(right)]);
      end
    end
end
ac=sum(acc)/(size(acc,2));
disp(['���ݼ���CDDE�㷨�ϲ��Ե�ƽ��׼ȷ��Ϊ��',num2str(ac)]);
toc;

