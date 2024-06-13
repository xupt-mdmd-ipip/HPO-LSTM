%% 初始化程序
warning off         % 关闭报警信息
close all               % 关闭所有图窗
clear                    % 清空变量
clc                        % 清空命令行
print_copr;           % 版权所有

%% 读取数据
sn = xlsread('数据集.xlsx');

%% 滑动时间窗重构数据集
delay=10;
for i=1:length(sn)-delay
    data(i,:)=sn(i:i+delay)';    % 构建时间窗内以及下一刻的样本序列
end

%% 划分训练集与测试集
% 重构数据集的总长度为500 - delay，即490
inputTrainDataset = data(1:390, 1:delay)';  % 前390行数据训练，时间窗长度内的数据作为输入变量
outputTrainDataset = data(1:390, delay + 1)';   % 前390行数据训练，时间窗长度外的下一刻为输出目标

inputTestDataset = data(391:490, 1:delay)'; % 391到490行数据训练，时间窗长度内的数据作为输入变量
outputTestDataset = data(391 : 490, delay + 1)';  % 391到490行数据训练，时间窗长度外的下一刻为输出目标

%% 归一化处理
[inputn_train, input_ps] = mapminmax(inputTrainDataset, 0, 1);
inputn_test = mapminmax('apply', inputTestDataset, input_ps);
[outputn_train, output_ps] = mapminmax(outputTrainDataset, 0, 1);

%% 设置参数
inputnode = length(inputn_train(:, 1)); % 输入层节点
outputnode = 1;  % 输出层节点

%% 调用优化算法
disp('running... ...(LSTM优化需要反复迭代吗，大概2个小时跑通)')
disp('初次跑代码，建议最大迭代次数改成5，种群数量改成4，来快速跑通程序.')
tic;
maxgen=50;   %最大迭代次数， 初次跑代码，建议最大迭代次数改成5，种群数量改成4，来快速跑通程序
popsize=30;   %种群数量
dim=4;    %变量维度，四个优化变量分别是LSTM的第一和第二隐含层节点数量，迭代循环数，以及初始学习率
lb=[10, 10, 100, 0.005];   %变量下界
ub=[300, 300, 800, 0.05];   %变量上界

% Constriction Coefeicient
B = 0.1;
curve = zeros(1,maxgen);
% 初始化种群位置和适应度
HPpos=rand(popsize,dim).*(ub-lb)+lb;
for i=1:size(HPpos,1)
    HPposFitness(i)=func(HPpos(i,:), inputnode, outputnode, inputn_train, outputn_train);
end
% 初始化最优个体位置和适应度
[~,indx] = min(HPposFitness);
bestx = HPpos(indx,:);   % Target HPO
bestf =HPposFitness(indx);
curve(1)=bestf;

%% 开始循环
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
% 结束优化

% 绘制优化算法的进化曲线
figure
plot(curve, 'r-', 'LineWidth', 1.0)
grid on
xlabel('进化代数')
ylabel('最佳适应度')
title('进化曲线')

%% 使用优化后的参数训练模型
hiddennode1 = round(bestx(1));  % 第一隐含层节点
hiddennode2 = round(bestx(2));  % 第二隐含层节点

%  创建网络
layers = [ ...
    sequenceInputLayer(inputnode)
    lstmLayer(hiddennode1,'OutputMode','last','name','hidden1')
    dropoutLayer(0.3,'name','dropout_1')                                     %隐藏层1权重丢失率，防止过拟合
    lstmLayer(hiddennode2,'OutputMode','last','name','hidden2')
    dropoutLayer(0.3,'name','dropout_2')                                    %隐藏层2权重丢失率，防止过拟合
    fullyConnectedLayer(outputnode,'name','fullconnect')
    regressionLayer('name','out')];             % %回归层

% 参数设定
%指定训练选项，
options = trainingOptions('adam', ...     %学习算法为adam
    'MaxEpochs',round(bestx(3)), ...         %遍历样本最大循环数
    'InitialLearnRate',bestx(4), ...              %初始学习率
    'LearnRateSchedule','piecewise', ...  % 学习率计划
    'LearnRateDropPeriod',50, ...         %50个epoch后学习率更新
    'LearnRateDropFactor',0.1, ...                  % 通过乘以因子 0.1 来降低学习率
    'MiniBatchSize',round(size(inputn_train, 2)/10),...             % 批处理样本大小每批次为训练集的十分之一样本
    'Verbose',1, ...        %命令控制台是否打印训练过程
    'Plots','training-progress');

%% 优化后的神经网络训练
net = trainNetwork(matToCell(inputn_train),outputn_train',layers,options);

%% 预测和反归一化
model_out1 = predict(net, matToCell(inputn_train));  % 训练集的归一化预测结果
model_out2 = predict(net, matToCell(inputn_test));    % 测试集的归一化预测结果
predictTrainDataset = mapminmax('reverse', model_out1', output_ps);  % 反归一化训练集预测结果为原始数量级
predictTestDataset = mapminmax('reverse', model_out2', output_ps);    % 反归一化测试集预测结果为原始数量级
toc;

%% 分析误差
disp('训练集误差计算如下: ')
MSE = mean((outputTrainDataset - predictTrainDataset).^2);
disp(['均方误差MSE = ', num2str(MSE)])
MAE = mean(abs(outputTrainDataset - predictTrainDataset));
disp(['平均绝对误差MAE = ', num2str(MAE)])
RMSE = sqrt(MSE);
disp(['根均方误差RMSE = ', num2str(RMSE)])
MAPE = mean(abs((outputTrainDataset - predictTrainDataset)./outputTrainDataset));
disp(['平均绝对百分比误差MAPE = ', num2str(MAPE*100), '%'])
R = corrcoef(outputTrainDataset, predictTrainDataset);
R2 = R(1, 2)^2;
disp(['拟合优度R2 = ', num2str(R2)])
disp(' ')
disp('测试集误差计算如下: ')
MSE_test = mean((outputTestDataset - predictTestDataset).^2);
disp(['均方误差MSE = ', num2str(MSE_test)])
MAE_test = mean(abs(outputTestDataset - predictTestDataset));
disp(['平均绝对误差MAE = ', num2str(MAE_test)])
RMSE_test = sqrt(MSE_test);
disp(['根均方误差RMSE = ', num2str(RMSE_test)])
MAPE_test = mean(abs((outputTestDataset - predictTestDataset)./outputTestDataset));
disp(['平均绝对百分比误差MAPE = ', num2str(MAPE_test*100), '%'])
R_test = corrcoef(outputTestDataset, predictTestDataset);
R2_test = R_test(1, 2)^2;
disp(['拟合优度R2 = ', num2str(R2_test)])

%% 对结果作图
% 训练集
figure
plot(outputTrainDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTrainDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('训练样本序号')
ylabel('目标')
legend('实际值', '预测值')
title({'HPO优化LSTM神经网络训练集预测值和实际值对比图', ['根均方误差RMSE = ', num2str(RMSE), '拟合优度R2 = ', num2str(R2)]})

figure
plot(outputTrainDataset - predictTrainDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('训练样本序号')
ylabel('预测偏差')
legend('误差')
title({'HPO优化LSTM神经网络训练集预测误差图', ['平均绝对百分比误差MAPE = ', num2str(MAPE*100), '%']})

% 测试集
figure
plot(outputTestDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTestDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('测试样本序号')
ylabel('目标')
legend('实际值', '预测值')
title({'HPO优化LSTM神经网络测试集预测值和实际值对比图', ['根均方误差RMSE = ', num2str(RMSE_test), '拟合优度R2 = ', num2str(R2_test)]})

figure
plot(outputTestDataset - predictTestDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('测试样本序号')
ylabel('预测偏差')
legend('误差')
title({'HPO优化LSTM神经网络测试集预测误差图', ['平均绝对百分比误差MAPE = ', num2str(MAPE_test*100), '%']})




