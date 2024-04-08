import torch
from examples.autogui.generate_data import gen_data
from examples.autogui.models import KeyMouseLSTM, ModelLoader

X, n_features = gen_data()
print("number of features:", n_features)

if __name__ == "__main__":
    model: KeyMouseLSTM = ModelLoader.load("keymouse-lstm")

    j = int(len(X) / model.batch_size) * model.batch_size
    # 提供待预测的数据
    input_data = torch.tensor(X[:j], dtype=torch.float32)

    # 将模型设置为评估模式
    model.eval()

    # 如果模型在GPU上训练，将数据移到GPU上
    if torch.cuda.is_available():
        model = model.cuda()
        input_data = input_data.cuda()

    # 使用no_grad()进行推理，减少内存消耗
    with torch.no_grad():
        predictions = model(input_data)

    # 将预测结果转换回CPU（如果在GPU上训练的话）并转换为numpy格式（如果需要的话）
    predictions = predictions.cpu().numpy()

    loss_function = torch.nn.MSELoss()  # 均方误差作为损失函数

    loss = loss_function(predictions, X[j+1])
    print(f"loss {loss.item():10.10f}")
