import torch
import torch.nn as nn
import numpy as np


# 定义RNN模型
class MouseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MouseRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class MouseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MouseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 假设鼠标轨迹序列为2D坐标(x, y)，每10个为一个序列
# 输入应该是一个[batch_size, sequence_length, input_size]的张量
mouse_sequences = np.random.rand(100, 10, 2)

# 转为张量
mouse_sequences_torch = torch.tensor(mouse_sequences, dtype=torch.float)

# 建立模型，input_size为2（2D坐标），hidden_size可以自定义，output_size为2（预测的下一个位置的2D坐标）
model = MouseLSTM(input_size=2, hidden_size=32, output_size=2)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(mouse_sequences_torch)
    # 计算损失
    loss = criterion(outputs, mouse_sequences_torch[:, -1, :])
    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 打印损失值
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


test_sequences = np.random.rand(20, 10, 2)  # 假设你有20组测试数据
test_sequences_torch = torch.tensor(test_sequences, dtype=torch.float)
with torch.no_grad():
    model.eval()
    outputs = model(test_sequences_torch)
    loss = criterion(outputs, test_sequences_torch[:, -1, :])
print("Test Loss:", loss.item())

# 假设我们有一段新的鼠标行为序列
new_sequence = np.random.rand(
    1, 10, 2
)  # 注意这里的形状，因为我们的模型接受的输入是3D的
new_sequence_torch = torch.tensor(new_sequence, dtype=torch.float)

# 使用模型进行预测
with torch.no_grad():
    model.eval()
    predicted_position = model(new_sequence_torch)

# 打印出预测的下一个位置
print("Predicted mouse position:", predicted_position.numpy())


def predict_future_positions(model, initial_sequence, future_steps):
    predicted_positions = []

    # 使用模型对给定的初始序列进行预测
    input_sequence = torch.tensor(initial_sequence, dtype=torch.float)
    with torch.no_grad():
        model.eval()
        predicted_position = model(input_sequence).numpy()
        predicted_positions.append(predicted_position)

    for i in range(future_steps - 1):
        # 将预测的位置加入到序列中，并且删除序列中的第一个元素，
        # 以保持序列的长度，然后用这个新的序列来预测下一个位置
        new_input_sequence = np.append(
            input_sequence[0][1:], predicted_position, axis=0
        )

        new_input_sequence_torch = torch.tensor(
            np.array([new_input_sequence]), dtype=torch.float
        )
        predicted_position = model(new_input_sequence_torch).detach().numpy()

        predicted_positions.append(predicted_position)

    return predicted_positions


# 假设我们有一段新的鼠标行为序列
new_sequence = np.random.rand(1, 10, 2)

result = predict_future_positions(model, new_sequence, 100)
for v in result:
    print(v[0])
