from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
from torch import nn

from examples.autogui.db import db
import numpy as np


def gen_data():
    df = db.read_data()

    # 将 timestamp 转化为seconds
    # 计算时间戳的均值和标准差
    timestamps = df["timestamp"]
    mean_timestamp = timestamps.mean()
    std_timestamp = timestamps.std()

    # 标准化时间戳
    normalized_timestamps = (timestamps - mean_timestamp) / std_timestamp

    df.fillna(
        {"key": "", "x": -1, "y": -1, "button": "", "pressed": False}, inplace=True
    )

    # 对 type 和 button 字段进行 one-hot 编码
    encoder = OneHotEncoder(max_categories=105)
    onehot_data = encoder.fit_transform(df[["type", "button", "key"]]).toarray()
    onehot_cols = encoder.get_feature_names_out(["type", "button", "key"])
    onehot_df = pd.DataFrame(onehot_data, columns=onehot_cols)

    # 将 x, y 坐标归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[["x", "y"]] = scaler.fit_transform(df[["x", "y"]])

    # 将 pressed 字段转化为 int
    df["pressed"] = df["pressed"].astype(int)

    # 合并 DataFrame
    processed_df = pd.concat(
        [normalized_timestamps, df[["x", "y", "pressed"]], onehot_df], axis=1
    )

    # print(processed_df.shape)
    # print(normalized_timestamps)
    # print(df[["x", "y", "pressed"]])
    # print(onehot_df.shape)

    # 将 processed_df 转化为 numpy array，并进行样本分割
    X = processed_df.to_numpy()

    _, n_features = X.shape

    return X, n_features


if __name__ == "__main__":
    X, n_features = gen_data()
    print(X)
