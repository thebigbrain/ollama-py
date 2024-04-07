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

n_steps = 3600
n_features = 6

X = gen_data(n_steps)
if __name__ == "__main__":
    model = ModelLoader.load("keymouse-lstm")
    print("out", model.predict(X))
