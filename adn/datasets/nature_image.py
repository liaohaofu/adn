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
from torchvision.transforms import CenterCrop, RandomCrop, Resize, RandomHorizontalFlip, Compose


class NatureImage(torch.utils.data.Dataset):
    def __init__(self, a_dir="data/train/nature_image/artifact", b_dir="data/train/nature_image/no_artifact",
        random_flip=True, load_size=384, crop_size=256, crop_type="random"):
        super(NatureImage, self).__init__()

        self.a_files = sorted(read_dir(a_dir, predicate="file", recursive=True))
        self.b_files = sorted(read_dir(b_dir, predicate="file", recursive=True))
        self.transform = Compose([
            Resize(load_size),
            {"center": CenterCrop, "random": RandomCrop}[crop_type](crop_size)
        ] + ([RandomHorizontalFlip()] if random_flip else []))
        

    def __len__(self):
        return len(self.a_files)

    def normalize(self, data):
        data = data / 255.0
        data = data * 2.0 - 1.0
        return data

    def to_tensor(self, data):
        if data.ndim == 2: data = data[np.newaxis, ...]
        elif data.ndim == 3: data = data.transpose(2, 0, 1)

        data = self.normalize(data)
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        data = self.denormalize(data)
        return data

    def denormalize(self, data):
        data = data * 0.5 + 0.5
        data = data * 255.0
        return data

    def get(self, a_file):
        data_name = path.basename(a_file)
        a = self.transform(Image.open(a_file).convert("RGB"))
        b = self.transform(Image.open(choice(self.b_files)).convert("RGB"))

        a = np.array(a).astype(np.float32)
        b = np.array(b).astype(np.float32)

        a = self.to_tensor(a)
        b = self.to_tensor(b)

        return {"data_name": data_name, "artifact": a, "no_artifact": b}

    def __getitem__(self, index):
        a_file = self.a_files[index]
        return self.get(a_file)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from adn.datasets import NatureImage
    dataset = NatureImage()
    data = dataset[100]

    a = dataset.to_numpy(data["artifact"]).astype(np.uint8)
    b = dataset.to_numpy(data["no_artifact"]).astype(np.uint8)

    plt.ion()
    plt.figure(); plt.imshow(a)
    plt.figure(); plt.imshow(b)