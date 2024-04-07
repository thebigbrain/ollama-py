from torch import nn
import torch

from examples.autogui.dataset import AutoGUIDataset
from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader, MultiTaskLSTM
from torch.utils.data import DataLoader

train_dataset, n_features = gen_data()

batch_size = 64  # 一个批次中的样本数量
train_data = AutoGUIDataset(train_dataset)  # train_dataset是实际的训练数据
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

input_size = n_features
hidden_size = 64
num_layers = 2
num_keys = 87

# 我们假设有 5 个训练周期
num_epochs = 5

# 初始化模型、优化器和损失函数
model = MultiTaskLSTM(input_size, hidden_size, num_layers, num_keys)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_position = nn.MSELoss()
criterion_type = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    for i, (actions, positions, types) in enumerate(train_loader):
        outputs_position, outputs_type = model(actions.float())

        loss_position = criterion_position(outputs_position, positions)
        loss_type = criterion_type(outputs_type, types)

        loss = loss_position + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


ModelLoader.save(model, "keymouse-lstm")
