from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch

from examples.autogui.db import db
import numpy as np

from examples.autogui.keys import keys_mapping
from xlab.preprocessing.screen import get_screen_scaler


def gen_data():
    df = db.read_data()

    # 将 timestamp 转化为seconds
    # 计算时间戳的均值和标准差
    timestamps = df["timestamp"]
    mean_timestamp = timestamps.mean()
    std_timestamp = timestamps.std()

    # 标准化时间戳
    normalized_timestamps = (timestamps - mean_timestamp) / std_timestamp

    df.fillna({"key": "", "x": 0, "y": 0, "button": "", "pressed": False}, inplace=True)

    # 对 type 和 button 字段进行 one-hot 编码
    encoder = OneHotEncoder()
    onehot_data = encoder.fit_transform(
        df[
            [
                "type",
                "button",
            ]
        ]
    ).toarray()
    onehot_cols = encoder.get_feature_names_out(
        [
            "type",
            "button",
        ]
    )
    onehot_df = pd.DataFrame(onehot_data, columns=onehot_cols)

    df["key"] = np.vectorize(lambda k: keys_mapping.get(k, 0))(df["key"])
    embedding = torch.nn.Embedding(len(keys_mapping), 20)
    key_data = embedding(torch.tensor(df["key"])).detach().numpy()
    key_df = pd.DataFrame(key_data)

    scaler = get_screen_scaler()
    df[["x", "y"]] = scaler.transform(df[["x", "y"]])

    # 将 pressed 字段转化为 int
    df["pressed"] = df["pressed"].astype(int)

    # 合并 DataFrame
    processed_df = pd.concat(
        [normalized_timestamps, df[["x", "y", "pressed"]], onehot_df, key_df],
        axis=1,
    )

    # 将 processed_df 转化为 numpy array，并进行样本分割
    X = processed_df.to_numpy()

    _, n_features = X.shape

    return X, n_features


if __name__ == "__main__":
    X, n_features = gen_data()
    print(X.shape)
