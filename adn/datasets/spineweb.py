import os
import os.path as path
import json
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from random import choice
from torch.utils.data import Dataset
from ..utils import read_dir


class Spineweb(torch.utils.data.Dataset):
    def __init__(self, a_dir="data/spineweb/train/artifact", b_dir="data/spineweb/train/no_artifact",
        random_flip=False, a_range=(-1000.0, 2000.0), b_range=(-1000.0, 2000.0)):
        super(Spineweb, self).__init__()

        self.a_files = sorted(read_dir(a_dir, predicate=lambda x: x.endswith("npy"), recursive=True))
        self.b_files = sorted(read_dir(b_dir, predicate=lambda x: x.endswith("npy"), recursive=True))

        self.random_flip = random_flip
        self.a_range = a_range
        self.b_range = b_range

    def __len__(self):
        return len(self.a_files)

    def normalize(self, data, minmax):
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 2.0 - 1.0
        return data

    def to_tensor(self, data, minmax):
        if self.random_flip and np.random.rand() > 0.5: data = data[:, ::-1]
        if data.ndim == 2: data = data[np.newaxis, ...]
        data = self.normalize(data, minmax)
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data, minmax=()):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        if minmax: data = self.denormalize(data, minmax)
        return data

    def denormalize(self, data, minmax):
        data_min, data_max = minmax
        data = data * 0.5 + 0.5
        data = data * (data_max - data_min) + data_min
        return data

    def get(self, a_file):
        data_name = path.basename(a_file)
        a = np.load(a_file).astype(np.float32)
        b = np.load(choice(self.b_files)).astype(np.float32)

        a = self.to_tensor(a, self.a_range)
        b = self.to_tensor(b, self.b_range)

        return {"data_name": data_name, "a": a, "b": b}

    def __getitem__(self, index):
        a_file = self.a_files[index]
        return self.get(a_file)
