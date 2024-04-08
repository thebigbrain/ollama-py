from torch import nn
import torch

from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader, KeyMouseLSTM

model_name = "keymouse-lstm"

batch_size = 64
train_dataset, n_features = gen_data(batch_size)

print("number of features:", n_features)

if __name__ == "__main__":
    model = ModelLoader.load(model_name)
    if model is None:
        print("first training: create model")
        model = KeyMouseLSTM(input_size=n_features, batch_size=batch_size)

    loss_function = nn.MSELoss()  # 均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化算法

    # 训练循环
    epochs = 150
    for i in range(epochs):
        for seq in train_dataset:
            optimizer.zero_grad()
            model.init_hidden()  # 加入这一行重置隐藏状态

            input_seq = torch.tensor([seq], dtype=torch.float32)

            y_predict = model(input_seq)

            single_loss = loss_function(y_predict, input_seq[0][-1])
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    ModelLoader.save(model, model_name)
