from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader

n_steps = 3600
n_features = 6

X = gen_data(n_steps)
if __name__ == '__main__':
    model = ModelLoader.load_model()
    print("out", model.predict(X))
