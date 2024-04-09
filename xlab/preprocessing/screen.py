from sklearn.preprocessing import FunctionTransformer
import numpy as np

# 假设您知道原始数据的最小值和最大值
min_x, max_x = 0, 1920
min_y, max_y = 0, 1080

delta = np.array([max_x, max_y]) - np.array([min_x, min_y])


# 自定义归一化函数
def custom_normalizer(X):
    X_scaled = (X - np.array([min_x, min_y])) / (delta)
    return X_scaled


# 自定义反归一化函数
def custom_inverse_normalizer(X_scaled):
    X = X_scaled * (delta) + np.array([min_x, min_y])
    return X


# 创建一个自定义的转换器


def get_screen_scaler(feature_names_out=None):
    transformer = FunctionTransformer(
        func=custom_normalizer,
        inverse_func=custom_inverse_normalizer,
        validate=True,
        feature_names_out=feature_names_out,
    )
    return transformer


if __name__ == "__main__":
    # 原始数据
    X = np.array([[1, 5], [2, 4], [3, 3], [4, 2000], [5, 1]])
    screen_transformer = get_screen_scaler()

    # 使用自定义转换器进行正向转换
    X_normalized = screen_transformer.transform(X)

    # 使用自定义转换器进行逆向转换
    X_original = screen_transformer.inverse_transform(X_normalized)

    # 打印结果
    print("Normalized data:")
    print(X_normalized)
    print("Original data:")
    print(X_original)
