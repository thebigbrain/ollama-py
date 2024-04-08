import torch
from examples.autogui.generate_data import gen_data
from examples.autogui.models import KeyMouseLSTM, ModelLoader

X, _ = gen_data()

if __name__ == "__main__":
    model: KeyMouseLSTM = ModelLoader.load("keymouse-lstm")

    # 提供待预测的数据
    input_data = torch.tensor(X[: int(len(X) / model.batch_size) * model.batch_size])

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

    print("out", predictions)
