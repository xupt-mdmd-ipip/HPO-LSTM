function error = func(x, inputnode,outputnode,inputn_train,outputn_train)
% �Ӻ������ڼ���˫������LSTM�Ż�����Ӧ��
hiddennode1 = round(x(1));  % ��һ������ڵ�
hiddennode2 = round(x(2));  % �ڶ�������ڵ�
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
    'MaxEpochs',round(x(3)), ...         %�����������ѭ����
    'InitialLearnRate',x(4), ...              %��ʼѧϰ��
    'LearnRateSchedule','piecewise', ...  % ѧϰ�ʼƻ�
    'LearnRateDropPeriod',50, ...         %50��epoch��ѧϰ�ʸ���
    'LearnRateDropFactor',0.1, ...                  % ͨ���������� 0.1 ������ѧϰ��
    'MiniBatchSize',round(size(inputn_train, 2)/10),...             % ������������Сÿ����Ϊѵ������ʮ��֮һ����
    'Verbose',1, ...        %�������̨�Ƿ��ӡѵ������
    'Plots','none');

% LSTM������ѵ��
net = trainNetwork(matToCell(inputn_train),outputn_train',layers,options);%����ѵ��
% ѵ�����Ĺ�һ������ֵ
model_out=predict(net, matToCell(inputn_train));
error=sqrt(mean((outputn_train - model_out').^2));



