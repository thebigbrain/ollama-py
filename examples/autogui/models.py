from keras.models import load_model as keras_load_model, Model
from os.path import isfile


class ModelLoader:
    @staticmethod
    def get_model_path():
        # 返回模型的存储路径
        return "model.keras"

    @staticmethod
    def load_model():
        model_path = ModelLoader.get_model_path()
        if isfile(model_path):
            # 如果模型存在，加载模型
            return keras_load_model(model_path)
        else:
            # 否则，返回None
            return None

    @staticmethod
    def save_model(model):
        # 保存模型到指定路径
        model.save(ModelLoader.get_model_path())


if __name__ == '__main__':
    pass
