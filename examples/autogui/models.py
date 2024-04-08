import os
from os.path import isfile

import torch
import torch.nn as nn

from xlab.core.resources import get_model_path


class KeyMouseLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, batch_size=1):
        super().__init__()
        self.hidden_cell = None
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, input_size)

        self.init_hidden()

    def init_hidden(self):
        self.hidden_cell = (
            torch.zeros(1, self.batch_size, self.hidden_layer_size),
            torch.zeros(1, self.batch_size, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), self.batch_size, self.lstm.input_size),
            self.hidden_cell,
        )
        predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))
        return predictions[-1]


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
        model_path = get_model_path(name)
        dir_path = os.path.dirname(model_path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(model, model_path)
