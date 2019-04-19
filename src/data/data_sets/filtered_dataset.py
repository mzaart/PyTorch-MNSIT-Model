import torch
from torch.utils.data import Dataset
from ...config import NUM_COLS


class FilteredDataSet(Dataset):

    def __init__(self, np_array, transform=None):
        self.transform = transform
        self.n = np_array.shape[0]
        np_pixels = np_array[:, :NUM_COLS-1]
        np_labels = np_array[:, NUM_COLS-1]
        self.pixels = torch.from_numpy(np_pixels)
        self.labels = torch.from_numpy(np_labels)

    def __getitem__(self, index):
        item = {
            'x': self.pixels[index],
            'y': self.labels[index]
        }
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return self.n
