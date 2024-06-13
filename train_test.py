import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from network import LSTM
import numpy as np
import pandas as pd

# 数据
from utils import grade

train_X, test_X, valid_X, train_y, test_y, valid_y = grade.loader_PM_10()
# train_X  (5502,24)
# train_y  5502
# 转成tensor
train_X = np.array(train_X)
test_X = np.array(test_X)
valid_X = np.array(valid_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
valid_y = np.array(valid_y)

train_X = torch.tensor(train_X)
train_y = torch.tensor(train_y)
valid_X = torch.tensor(valid_X)
test_X = torch.tensor(test_X)
test_y = torch.tensor(test_y)
valid_y = torch.tensor(valid_y)

batch_size = 1442
train = torch.utils.data.TensorDataset(train_X, train_y)
test = torch.utils.data.TensorDataset(test_X, test_y)
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

input_dim = 24  # 数据的特征数
hidden_dim = 100  # 隐藏层的神经元个数
num_layers = 3  # LSTM的层数
output_dim = 1  # 预测值的特征数

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss(size_average=True)
num_epochs = 5000
hist = np.zeros(num_epochs)
total_loss = 0
best_loss = 10
for t in range(num_epochs):
    for index in range(0, train_X.shape[0], batch_size):
        x = train_X[index:index + batch_size, :]
        y = train_y[index:index + batch_size]
        x = x.resize(x.size(0), 1, 24).float()
        y = y.resize(x.size(0), 1, 1).float()
        y_train_pred = model(x)
        loss = loss_fn(y_train_pred.float(), train_y.float())
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        total_loss += loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    loss = total_loss / train_X.shape[0]
    torch.save(model.state_dict(), './weights/LSTM_weight_PM_10_train.pkl')
    print(f"epoch={t}|_train_loss={loss}\n\n")

    # types = ['PM10(μg/m3)', 'PM2.5(μg/m3)', 'SO2(μg/m3)', 'NO2(μg/m3)', 'CO(mg/m3)', 'O3(μg/m3)']
    # models = [model_LSTM]
    # models_name = ['LSTM']
    #
    # circle = 100
    # batch_size = 512
    # gru_loss = pd.DataFrame(columns=types, index=range(circle))
    # print(gru_loss)
    # for i in range(len(models)):
    #     models[i].train()
    #     best_loss = [1000, 1000, 1000, 1000, 1000, 1000]
    #     # 训练
    #     print(f'start train {models_name[i]}\n\n')
    #     for type in range(len(types)):
    #         print(f'************start train {types[type]}')
    #         for epoch in range(circle):
    #
    #             models[i].train()
    #             total_correct = 0
    #             total_loss = 0
    #             acc = 0
    #             for index in range(0, train_X.shape[0], batch_size):
    #                 x = train_X[index:index + batch_size, :]
    #                 y = train_y[index:index + batch_size, type]
    #                 x = x.resize(x.size(0), 1, 144).float()
    #                 y = y.resize(x.size(0), 1, 1).float()
    #                 pred = model_LSTM(x)
    #
    #                 loss = criterion(pred, y)
    #                 total_loss += loss
    #                 print(pred)
    #
    #                 loss.backward()
    #
    #                 optimizer_LSTM.step()
    #                 optimizer_LSTM.zero_grad()
    #             loss = total_loss / train_X.shape[0]
    #
    #             torch.save(models[i].state_dict(),
    #                        'F:/Jetbrains/python/HPO_LSTM/weights/{}_weight_{}_train.pkl'.format(models_name[i], types[type][0:-7]))
    #             print(f"epoch={epoch}|{types[type][0:-7]}_train_loss={loss}\n\n")
    #
    # 测试
    model.eval()
    for epoch_1 in range(1):

        total_correct = 0
        total_loss = 0
        for index in range(0, valid_X.shape[0], batch_size):
            x = valid_X[index:index + batch_size, :]
            y = valid_y[index:index + batch_size]
            x = x.resize(x.size(0), 1, 24).float()
            y = y.resize(x.size(0), 1, 1).float()

            pred = model(x)
            loss = loss_fn(pred.float(), y.float())

            correct = torch.sum(pred == y)
            total_correct += correct
            total_loss += loss

        loss = total_loss / valid_X.shape[0]

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),
                       './weights/LSTM_weight_PM_10_valid.pkl')
        print(f"valid_loss={loss}|best_loss={best_loss}")
print('\n*********************\n')
