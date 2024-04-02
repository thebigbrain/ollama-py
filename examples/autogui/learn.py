from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.optimizers import Adam
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau

from examples.autogui.generate_data import gen_data
from examples.autogui.models import ModelLoader

n_steps = 3600
activation = 'tanh'

X, n_features = gen_data(n_steps)

inputs = Input(shape=(n_steps, n_features))
encoded = LSTM(50, activation=activation)(inputs)

# Repeat the encoded output n_steps times to match the decoder structure
repeated_out = RepeatVector(n_steps)(encoded)

# Decoder structure
decoded = LSTM(n_features, return_sequences=True, activation=activation)(repeated_out)

if __name__ == '__main__':
    # 尝试加载已存在的模型
    autoencoder = ModelLoader.load_model()

    if autoencoder is None:
        # 如果不存在模型，构造一个新的autoencoder模型
        autoencoder = Model(inputs, decoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    # 定义一个检查点（checkpoint）回调，保存在训练过程中验证集损失最小的模型
    checkpoint_cb = ModelCheckpoint(ModelLoader.get_model_path(), monitor='val_loss', save_best_only=True, mode='min')

    # Training.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001)
    autoencoder.fit(X, X, epochs=200, validation_split=0.2, callbacks=[checkpoint_cb, reduce_lr])
    print(autoencoder.summary())

    # Save the improved model
    ModelLoader.save_model(autoencoder)
