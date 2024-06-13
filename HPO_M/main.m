%% ��ʼ������
warning off         % �رձ�����Ϣ
close all               % �ر�����ͼ��
clear                    % ��ձ���
clc                        % ���������
print_copr;           % ��Ȩ����

%% ��ȡ����
sn = xlsread('���ݼ�.xlsx');

%% ����ʱ�䴰�ع����ݼ�
delay=10;
for i=1:length(sn)-delay
    data(i,:)=sn(i:i+delay)';    % ����ʱ�䴰���Լ���һ�̵���������
end

%% ����ѵ��������Լ�
% �ع����ݼ����ܳ���Ϊ500 - delay����490
inputTrainDataset = data(1:390, 1:delay)';  % ǰ390������ѵ����ʱ�䴰�����ڵ�������Ϊ�������
outputTrainDataset = data(1:390, delay + 1)';   % ǰ390������ѵ����ʱ�䴰���������һ��Ϊ���Ŀ��

inputTestDataset = data(391:490, 1:delay)'; % 391��490������ѵ����ʱ�䴰�����ڵ�������Ϊ�������
outputTestDataset = data(391 : 490, delay + 1)';  % 391��490������ѵ����ʱ�䴰���������һ��Ϊ���Ŀ��

%% ��һ������
[inputn_train, input_ps] = mapminmax(inputTrainDataset, 0, 1);
inputn_test = mapminmax('apply', inputTestDataset, input_ps);
[outputn_train, output_ps] = mapminmax(outputTrainDataset, 0, 1);

%% ���ò���
inputnode = length(inputn_train(:, 1)); % �����ڵ�
outputnode = 1;  % �����ڵ�

%% �����Ż��㷨
disp('running... ...(LSTM�Ż���Ҫ���������𣬴��2��Сʱ��ͨ)')
disp('�����ܴ��룬���������������ĳ�5����Ⱥ�����ĳ�4����������ͨ����.')
tic;
maxgen=50;   %������������ �����ܴ��룬���������������ĳ�5����Ⱥ�����ĳ�4����������ͨ����
popsize=30;   %��Ⱥ����
dim=4;    %����ά�ȣ��ĸ��Ż������ֱ���LSTM�ĵ�һ�͵ڶ�������ڵ�����������ѭ�������Լ���ʼѧϰ��
lb=[10, 10, 100, 0.005];   %�����½�
ub=[300, 300, 800, 0.05];   %�����Ͻ�

% Constriction Coefeicient
B = 0.1;
curve = zeros(1,maxgen);
% ��ʼ����Ⱥλ�ú���Ӧ��
HPpos=rand(popsize,dim).*(ub-lb)+lb;
for i=1:size(HPpos,1)
    HPposFitness(i)=func(HPpos(i,:), inputnode, outputnode, inputn_train, outputn_train);
end
% ��ʼ�����Ÿ���λ�ú���Ӧ��
[~,indx] = min(HPposFitness);
bestx = HPpos(indx,:);   % Target HPO
bestf =HPposFitness(indx);
curve(1)=bestf;

%% ��ʼѭ��
for it = 2:maxgen
    c = 1 - it*((0.98)/maxgen);   % Update C Parameter
    kbest=ceil(popsize*c);        % Update kbest
    for i = 1:popsize
        r1=rand(1,dim)<c;
        r2=rand;
        r3=rand(1,dim);
        idx=(r1==0);
        z=r2.*idx+r3.*~idx;
        if rand<B
            xi=mean(HPpos);
            dist = pdist2(xi,HPpos);
            [~,idxsortdist]=sort(dist);
            SI=HPpos(idxsortdist(kbest),:);
            HPpos(i,:) =HPpos(i,:)+0.5*((2*(c)*z.*SI-HPpos(i,:))+(2*(1-c)*z.*xi-HPpos(i,:)));
        else
            for j=1:dim
                rr=-1+2*z(j);
                HPpos(i,j)= 2*z(j)*cos(2*pi*rr)*(bestx(j)-HPpos(i,j))+bestx(j);
            end
        end
        HPpos(i,:) = min(max(HPpos(i,:),lb),ub);
        % Evaluation
        HPposFitness(i) = func(HPpos(i,:), inputnode, outputnode, inputn_train, outputn_train);
        % Update Target
        if HPposFitness(i)<bestf
            bestx = HPpos(i,:);
            bestf = HPposFitness(i);
        end
    end
    curve(it)=bestf;
end
% �����Ż�

% �����Ż��㷨�Ľ�������
figure
plot(curve, 'r-', 'LineWidth', 1.0)
grid on
xlabel('��������')
ylabel('�����Ӧ��')
title('��������')

%% ʹ���Ż���Ĳ���ѵ��ģ��
hiddennode1 = round(bestx(1));  % ��һ������ڵ�
hiddennode2 = round(bestx(2));  % �ڶ�������ڵ�

%  ��������
layers = [ ...
    sequenceInputLayer(inputnode)
    lstmLayer(hiddennode1,'OutputMode','last','name','hidden1')
    dropoutLayer(0.3,'name','dropout_1')                                     %���ز�1Ȩ�ض�ʧ�ʣ���ֹ�����
    lstmLayer(hiddennode2,'OutputMode','last','name','hidden2')
    dropoutLayer(0.3,'name','dropout_2')                                    %���ز�2Ȩ�ض�ʧ�ʣ���ֹ�����
    fullyConnectedLayer(outputnode,'name','fullconnect')
    regressionLayer('name','out')];             % %�ع��

% �����趨
%ָ��ѵ��ѡ�
options = trainingOptions('adam', ...     %ѧϰ�㷨Ϊadam
    'MaxEpochs',round(bestx(3)), ...         %�����������ѭ����
    'InitialLearnRate',bestx(4), ...              %��ʼѧϰ��
    'LearnRateSchedule','piecewise', ...  % ѧϰ�ʼƻ�
    'LearnRateDropPeriod',50, ...         %50��epoch��ѧϰ�ʸ���
    'LearnRateDropFactor',0.1, ...                  % ͨ���������� 0.1 ������ѧϰ��
    'MiniBatchSize',round(size(inputn_train, 2)/10),...             % ������������Сÿ����Ϊѵ������ʮ��֮һ����
    'Verbose',1, ...        %�������̨�Ƿ��ӡѵ������
    'Plots','training-progress');

%% �Ż����������ѵ��
net = trainNetwork(matToCell(inputn_train),outputn_train',layers,options);

%% Ԥ��ͷ���һ��
model_out1 = predict(net, matToCell(inputn_train));  % ѵ�����Ĺ�һ��Ԥ����
model_out2 = predict(net, matToCell(inputn_test));    % ���Լ��Ĺ�һ��Ԥ����
predictTrainDataset = mapminmax('reverse', model_out1', output_ps);  % ����һ��ѵ����Ԥ����Ϊԭʼ������
predictTestDataset = mapminmax('reverse', model_out2', output_ps);    % ����һ�����Լ�Ԥ����Ϊԭʼ������
toc;

%% �������
disp('ѵ��������������: ')
MSE = mean((outputTrainDataset - predictTrainDataset).^2);
disp(['�������MSE = ', num2str(MSE)])
MAE = mean(abs(outputTrainDataset - predictTrainDataset));
disp(['ƽ���������MAE = ', num2str(MAE)])
RMSE = sqrt(MSE);
disp(['���������RMSE = ', num2str(RMSE)])
MAPE = mean(abs((outputTrainDataset - predictTrainDataset)./outputTrainDataset));
disp(['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE*100), '%'])
R = corrcoef(outputTrainDataset, predictTrainDataset);
R2 = R(1, 2)^2;
disp(['����Ŷ�R2 = ', num2str(R2)])
disp(' ')
disp('���Լ�����������: ')
MSE_test = mean((outputTestDataset - predictTestDataset).^2);
disp(['�������MSE = ', num2str(MSE_test)])
MAE_test = mean(abs(outputTestDataset - predictTestDataset));
disp(['ƽ���������MAE = ', num2str(MAE_test)])
RMSE_test = sqrt(MSE_test);
disp(['���������RMSE = ', num2str(RMSE_test)])
MAPE_test = mean(abs((outputTestDataset - predictTestDataset)./outputTestDataset));
disp(['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE_test*100), '%'])
R_test = corrcoef(outputTestDataset, predictTestDataset);
R2_test = R_test(1, 2)^2;
disp(['����Ŷ�R2 = ', num2str(R2_test)])

%% �Խ����ͼ
% ѵ����
figure
plot(outputTrainDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTrainDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('ѵ���������')
ylabel('Ŀ��')
legend('ʵ��ֵ', 'Ԥ��ֵ')
title({'HPO�Ż�LSTM������ѵ����Ԥ��ֵ��ʵ��ֵ�Ա�ͼ', ['���������RMSE = ', num2str(RMSE), '����Ŷ�R2 = ', num2str(R2)]})

figure
plot(outputTrainDataset - predictTrainDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('ѵ���������')
ylabel('Ԥ��ƫ��')
legend('���')
title({'HPO�Ż�LSTM������ѵ����Ԥ�����ͼ', ['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE*100), '%']})

% ���Լ�
figure
plot(outputTestDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTestDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('�����������')
ylabel('Ŀ��')
legend('ʵ��ֵ', 'Ԥ��ֵ')
title({'HPO�Ż�LSTM��������Լ�Ԥ��ֵ��ʵ��ֵ�Ա�ͼ', ['���������RMSE = ', num2str(RMSE_test), '����Ŷ�R2 = ', num2str(R2_test)]})

figure
plot(outputTestDataset - predictTestDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('�����������')
ylabel('Ԥ��ƫ��')
legend('���')
title({'HPO�Ż�LSTM��������Լ�Ԥ�����ͼ', ['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE_test*100), '%']})




