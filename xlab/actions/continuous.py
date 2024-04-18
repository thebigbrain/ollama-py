import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型
class ContinuousActionModel(nn.Module):
    def __init__(self):
        super(ContinuousActionModel, self).__init__()
        self.fc = nn.Linear(
            in_features=10, out_features=2
        )  # 假设状态空间维度为10，动作空间维度为2

    def forward(self, x):
        return self.fc(x)


# 实例化模型、优化器
model = ContinuousActionModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()  # 均方误差作为损失函数


# 示范一个训练步骤
def train_step(state, action, reward):
    predicted_action = model(state)
    loss = -reward * loss_function(predicted_action, action)  # 定义损失函数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# some_loss_function应由你根据实际任务来定义
# 比如可以是动作的均方误差（MSE），如果动作是多维的，则可能需要计算每一维的MSE
