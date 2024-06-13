function error = func(x, inputnode,outputnode,inputn_train,outputn_train)
% 子函数用于计算双隐含层LSTM优化的适应度
hiddennode1 = round(x(1));  % 第一隐含层节点
hiddennode2 = round(x(2));  % 第二隐含层节点
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
    'MaxEpochs',round(x(3)), ...         %遍历样本最大循环数
    'InitialLearnRate',x(4), ...              %初始学习率
    'LearnRateSchedule','piecewise', ...  % 学习率计划
    'LearnRateDropPeriod',50, ...         %50个epoch后学习率更新
    'LearnRateDropFactor',0.1, ...                  % 通过乘以因子 0.1 来降低学习率
    'MiniBatchSize',round(size(inputn_train, 2)/10),...             % 批处理样本大小每批次为训练集的十分之一样本
    'Verbose',1, ...        %命令控制台是否打印训练过程
    'Plots','none');

% LSTM神经网络训练
net = trainNetwork(matToCell(inputn_train),outputn_train',layers,options);%网络训练
% 训练集的归一化仿真值
model_out=predict(net, matToCell(inputn_train));
error=sqrt(mean((outputn_train - model_out').^2));



