import numpy as np
from torch.utils.data import Dataset


class ShuffledDataSet(Dataset):

    def __init__(self, data_set):
        self.data_set = data_set
        self.indices = np.random.permutation(len(data_set))

    def __getitem__(self, index):
        return self.data_set[self.indices[index]]

    def __len__(self):
        return len(self.data_set)
