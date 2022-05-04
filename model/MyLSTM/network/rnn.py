import torch
import torch.nn as nn

feature_size = 169  # 如果是单幅图片分类，就是每行像素点数；如果是序列图像，则是
output_size = 10  # 10分类问题
batch_size = 4
hidden_size = 32


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, 0, 0.001)  # 参数初始化
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        out, hidden_prev = self.rnn(x, h_prev)
        out = self.linear(out[:, -1, :])
        return out, hidden_prev
