from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd

from examples.autogui.db import db


def gen_data(n_steps=3600):
    df = db.read_data()

    # 将 timestamp 转化为seconds
    start_time = df["timestamp"].min()
    df["timestamp"] = (df["timestamp"] - start_time).dt.total_seconds()

    df.fillna(
        {"key": "", "x": -1, "y": -1, "button": "", "pressed": False}, inplace=True
    )

    # 对 type 和 button 字段进行 one-hot 编码
    encoder = OneHotEncoder()
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
        [df["timestamp"], df[["x", "y", "pressed"]], onehot_df], axis=1
    )

    # 将 processed_df 转化为 numpy array，并进行样本分割
    X = processed_df.to_numpy()
    n_samples = int(len(X) / n_steps)

    X = X[: n_samples * n_steps].reshape(n_samples, n_steps, -1)

    _, _, n_features = X.shape

    return X, n_features
