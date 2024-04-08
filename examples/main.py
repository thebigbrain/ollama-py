import torch
from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader

"""
event = {
        'timestamp': datetime.now(),
        'type': 'mouse_scroll',
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy,
        'key': str(key)
    }
"""

X, _ = gen_data()

if __name__ == "__main__":
    model = ModelLoader.load("keymouse-lstm")
    print(X)

    # 提供待预测的数据
    input_data = torch.tensor(X)

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

    model = ModelLoader.load("keymouse-lstm")
    print("out", model.predict(X))
