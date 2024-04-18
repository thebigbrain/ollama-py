import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型
class MouseActionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MouseActionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh(),  # 输出值在[-1, 1]之间
        )

    def forward(self, x):
        return self.network(x)


# 实例化模型
input_size = 1024  # 假设的状态向量大小
output_size = 2  # x和y方向的移动
model = MouseActionNN(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # 以均方误差作为损失函数

if __name__ == "__main__":
    # 假设的状态和动作示例（请替换成你的环境逻辑）
    dummy_states = torch.rand((10, input_size))  # 假设有10个样本
    dummy_actions = torch.rand((10, output_size)) * 2 - 1  # 动作在[-1, 1]之间

    # 训练模型
    for epoch in range(100):  # 使用100个epochs进行示例
        # 在实际应用中，你需要根据环境逻辑获取状态和正确的动作/奖励
        pred_actions = model(dummy_states)
        loss = loss_fn(pred_actions, dummy_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
