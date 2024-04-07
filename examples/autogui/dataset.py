from torch.utils.data import Dataset


class AutoGUIDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        # 返回数据集的大小
        return len(self.X)

    def __getitem__(self, index):
        # 返回一条数据
        return self.X[index]
