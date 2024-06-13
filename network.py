import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# if __name__ == '__main__':
#     input_dim = 24  # 数据的特征数
#     hidden_dim = 24  # 隐藏层的神经元个数
#     layer_dim = 3  # LSTM的层数
#     output_dim = 1  # 预测值的特征数
#
#     lstm = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
#     print(lstm(torch.Tensor(32, 1, 24)).size())
