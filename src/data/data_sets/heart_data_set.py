import torch
from torch.utils.data import Dataset
from ...config import NUM_COLS


class HeartDataSet(Dataset):
    """
    Loads the UCI Heart Disease data set.
    """

    def __init__(self, np_array, transform=None):
        self.transform = transform
        self.n = np_array.shape[0]
        np_features = np_array[:, :NUM_COLS-1]
        np_labels = np_array[:, NUM_COLS-1]
        self.features = torch.from_numpy(np_features).float()
        self.labels = torch.from_numpy(np_labels).float()

    def __getitem__(self, index):
        item = {
            'x': self.features[index],
            'y': self.labels[index]
        }
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return self.n
