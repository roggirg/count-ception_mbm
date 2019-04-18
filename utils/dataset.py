import torch

from torch.utils import data
import pickle
import numpy as np


class MBM(data.Dataset):
    def __init__(self, pkl_file, transform, mode='train'):
        self.dataset = pickle.load(open(pkl_file, "rb"))
        self.transform = transform
        self.mode = mode

        self.create_xyc()
        self.split_data()

    def create_xyc(self):
        self.np_dataset_x = np.asarray([d[0] for d in self.dataset])
        self.np_dataset_y = np.asarray([d[1] for d in self.dataset])
        self.np_dataset_c = np.asarray([d[2] for d in self.dataset])
        self.np_dataset_x = self.np_dataset_x.transpose((0, 3, 1, 2))

    def split_data(self):
        n = 15
        if self.mode == 'train':
            self.x = self.np_dataset_x[0:n]
            self.y = self.np_dataset_y[0:n]
            self.c = self.np_dataset_c[0:n]
            print("np_dataset_x_train", len(self.x))

        elif self.mode == 'valid':
            self.x = self.np_dataset_x[n:2 * n]
            self.y = self.np_dataset_y[n:2 * n]
            self.c = self.np_dataset_c[n:2 * n]
            print("np_dataset_x_valid", len(self.x))

        else:
            self.x = self.np_dataset_x[-11:]
            self.y = self.np_dataset_y[-11:]
            self.c = self.np_dataset_c[-11:]
            print("np_dataset_x_test", len(self.x))

    def __getitem__(self, index):
        img = self.x[index]
        label = self.y[index]
        count = self.c[index]
        return torch.from_numpy(img).float(), torch.from_numpy(label).float(), count

    def __len__(self):
        """Return the number of images."""
        return len(self.x)
