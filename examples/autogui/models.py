from os.path import isfile

import torch

from xlab.core.resources import get_model_path

import torch.nn as nn


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MultiTaskLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 预测鼠标的位置和动作类型
        self.fc_position = nn.Linear(hidden_size, 2)
        self.fc_type = nn.Linear(hidden_size, 5)  # 有5种可能的动作，包括'key_press'

        # 预测键盘输入
        self.fc_key = nn.Linear(hidden_size, 87)  # num_keys是可能的输入字符数量

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out_position = self.fc_position(out)
        out_type = self.fc_type(out)
        out_key = self.fc_key(out)
        return out_position, out_type, out_key


class ModelLoader:
    @staticmethod
    def load(name):
        model_path = get_model_path(name)
        if isfile(model_path):
            # 如果模型存在，加载模型
            return torch.load(model_path)
        else:
            # 否则，返回None
            return None

    @staticmethod
    def save(model, name):
        # 保存模型到指定路径
        torch.save(model, get_model_path(name))
