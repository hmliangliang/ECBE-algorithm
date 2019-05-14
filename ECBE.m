cu=SEA;%data set is SEA
col=size(cu,2);%数据集的维度
data=cu(:,1:(col-1));
target=cu(:,col);
klabel=max(unique(target));%数据集的类标签数目
winsize=100;
a=0.05;%显著性水平
b=2;%子分类器更新权值的增加因子，b>1
r=0.1;%发生概念漂移后新加入分类器的权值的相乘因子
j=0;%记录当前分类器的个数
k=5;%系统规定子分类器的阀值
s=[];%记录每个子分类器对当前数据集的分类结果的熵值
theta=2*sqrt((log2(1/a))/(2000*100));
theta_weight=0;
tt=0;%记录当前测试的次数
acc=[];%记录每一次测试的准确率
detaH1=0;
detaH2=detaH1;
ensemble=struct('traindata',[],'traintarget',[],'weight',[]);
tic;
for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %形成训练集
        traindata=data((i-2*winsize+1):(i-winsize),:);
        traintarget=target((i-2*winsize+1):(i-winsize),:);
        if j<k %系统中的分类器不足
            if j==0 %系统中无分类器
                ensemble(1).traindata=traindata;
                ensemble(1).traintarget=traintarget;
                ensemble(1).weight=1;
                j=j+1;
            else %系统中有分类器但未达到上限
                temp=struct('traindata',[],'traintarget',[],'weight',[]);
                temp.traindata=traindata;
                temp.traintarget=traintarget;
                temp.weight=1;
                ensemble=[ensemble,temp];
                j=j+1;
            end
        end
      if j~=0
            H1=Entropy(traintarget);%计算当前数据块的熵值
            result=[];%记录每个分类器对训练集的分类结果
            for ei=1:size(ensemble,2)%用当前分类器对traindata进行分类
                model=ClassificationTree.fit(ensemble(ei).traindata,ensemble(ei).traintarget);
                res=predict(model,traindata);
                h=Entropy(res);%子分类器的分类结果的熵值
                s=[s,h];%保存每一个子分类器分类结果的熵值
                ch=abs(h-H1);%熵的变化情况
                %更新子分类器的权值
                if ch>0 %有分类错误的情况，衰减权值
                   ensemble(ei).weight=(1/(exp(1+ch)))*(ensemble(ei).weight);
                else %完全分类正确，增加权值，b>1
                   ensemble(ei).weight=b*(ensemble(ei).weight);
                end
                result=[result,res];%行代表实例，列代表分类器
            end
            %根据分类结果加权投票作出决策
            wr=zeros(size(traindata,1),klabel);%记录投票的权值状况
            for ex=1:size(result,1)%统计投票情况
                for ey=1:size(result,2)
                    wr(ex,result(ex,ey))=wr(ex,result(ex,ey))+ensemble(ey).weight;
                end
            end
            [c,last]=max(wr,[],2);%依据加权结果作出决策，last记录决策结果
            H2=Entropy(last);%H2代表分类后的数据集的熵值
            detaH2=abs(H2-H1);
            if tt==0
                detaH1=detaH2;
            end
            %判断是否发生概念漂移
            if (abs(detaH1-detaH2)>theta)||(detaH1>theta)||(detaH2>theta) %发生了概念漂移
                disp('概念漂移');
                [c,d]=min([ensemble.weight]);%选出权值最小的分类器
                ensemble(d)=[];%删除
                j=j-1;
                if size(s,2)>1
                    u=mean(s);%求子分类器熵值的均值u
                    v=std(s);%求子分类器熵值的标准差v
                    theta_weight=u-(v/(sqrt(size(s,2)))*tinv(0.95,(size(s,2)-1)))-3*v;
                    s=[];
                else
                    theta_weight=0;
                end
                ef=1;
                while ef<=size(ensemble,2)%对当前分类器系统之中，权值低于theta_weight的子分类器都要删除
                    if ensemble(ef).weight<theta_weight
                       ensemble(ef)=[];%删除分类器
                       j=j-1;
                    else
                       ef=ef+1;%当前分类器符合要求，继续向前扫描
                    end
                end
                %创建新分类器
                temp1=struct('traindata',[],'traintarget',[],'weight',[]);
                temp1.weight=r*sum([ensemble.weight]);
                if j<k %分类器未达到上限
                    ensemble=[ensemble,temp1];%将新训练的分类器加入到系统中去
                else
                    [c,d]=min([ensemble.weight]);%选出权值最小的分类器
                    ensemble(d)=[];%删除
                    ensemble=[ensemble,temp1];
                end
            end
            detaH1=detaH2;
            for qa=1:size(ensemble,2)%用训练集去训练每一个分类器
                ensemble(qa).traindata=[ensemble(qa).traindata;traindata];
                ensemble(qa).traintarget=[ensemble(qa).traintarget;traintarget];
            end
            %获得测试集
            tt=tt+1;
            testdata=data((i-winsize+1):i,:);
            testtarget=target((i-winsize+1):i,:);
            cresult=[];%记录每个子分类器对测试集的分类情况
            for di=1:size(ensemble,2)
                cmodel=ClassificationTree.fit(ensemble(di).traindata,ensemble(di).traintarget);%获得分类模型
                cres=predict(cmodel,testdata);
                cresult=[cresult,cres];%行代表实例，列代表分类器
            end
            cw=zeros(size(testdata,1),klabel);%cw记录各个实例在每个类标签上的投票权值
            for x=1:size(cresult,1)
                for y=1:size(cresult,2)
                    cw(x,cresult(x,y))=cw(x,cresult(x,y))+ensemble(y).weight;
                end
            end
            [c,clabel]=max(cw,[],2);%依照权重最大原则作出决策
            %统计正确分类实例个数
            count=0;
            for g=1:size(clabel,1)
                if clabel(g,1)==testtarget(g,1)
                    count=count+1;
                end
            end
            right=count/(size(clabel,1));%准确率
            acc=[acc,right];
            disp(['第',num2str(tt),'次测试的正确率为：',num2str(right)]);
      end
    end
end
ac=sum(acc)/(size(acc,2));
disp(['数据集在CDDE算法上测试的平均准确率为：',num2str(ac)]);
toc;

