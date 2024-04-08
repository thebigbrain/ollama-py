from torch import nn
import torch
import numpy as np

from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader, KeyMouseLSTM

model_name = "keymouse-lstm"
batch_size = 64

train_dataset, n_features = gen_data()

print("number of features:", n_features)

if __name__ == "__main__":
    model: KeyMouseLSTM = ModelLoader.load(model_name)
    if model is None or model.lstm.input_size != n_features:
        print(f"create model {model_name}")
        model = KeyMouseLSTM(input_size=n_features, batch_size=batch_size)

    loss_function = nn.MSELoss()  # 均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化算法

    # 训练循环
    epochs = 150
    for i in range(epochs):
        j = 0
        while j < len(train_dataset) - batch_size - 1:
            optimizer.zero_grad()
            model.init_hidden()  # 加入这一行重置隐藏状态

            seq = train_dataset[j : j + batch_size]
            j = j + batch_size
            input_seq = torch.tensor(np.array([seq]), dtype=torch.float32)

            y_predict = model(input_seq)

            single_loss = loss_function(
                y_predict, torch.tensor(train_dataset[j + 1], dtype=torch.float)
            )
            single_loss.backward()
            optimizer.step()

        print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

        if i > 0 and i % 33 == 0:
            ModelLoader.save(model, model_name)

    ModelLoader.save(model, model_name)
