import json
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from config import config
import numpy as np


class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms."""

    def __init__(self, data_file):
        tensors = torch.from_numpy(np.load(data_file, allow_pickle=True))
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x / 255.0 - 1.0),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)
