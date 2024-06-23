import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def caltech101_transformer():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])

class Caltech101(Dataset):
    def __init__(self, root_path):
        self.root = os.path.join(root_path, 'train')
        if not os.path.exists(self.root):
            raise RuntimeError(f'Dataset not found at {self.root}. Please download manually.')

        self.transform = caltech101_transformer()
        self._load_dataset()

    def _load_dataset(self):
        self.caltech101 = datasets.ImageFolder(self.root, transform=self.transform)

    def __getitem__(self, index):
        return self.caltech101[index]

    def __len__(self):
        return len(self.caltech101)

class rot_Caltech101(Dataset):
    def __init__(self, root_path):
        self.root = os.path.join(root_path, 'train')
        if not os.path.exists(self.root):
            raise RuntimeError(f'Dataset not found at {self.root}. Please download manually.')

        self.transform = caltech101_transformer()
        self._load_dataset()

    def _load_dataset(self):
        self.caltech101 = datasets.ImageFolder(self.root, transform=self.transform)

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 180 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, index):
        data, target = self.caltech101[index]
        data = np.asarray(data)

        data = np.moveaxis(data, 0, -1)
        data0 = data
        data90 = self.rotate_img(data, 90)
        data180 = self.rotate_img(data, 180)
        data270 = self.rotate_img(data, 270)

        data = np.moveaxis(data, 2, 0)
        data0 = np.moveaxis(data0, 2, 0)
        data90 = np.moveaxis(data90, 2, 0)
        data180 = np.moveaxis(data180, 2, 0)
        data270 = np.moveaxis(data270, 2, 0)

        data0 = torch.from_numpy(data0.copy()).float()
        data90 = torch.from_numpy(data90.copy()).float()
        data180 = torch.from_numpy(data180.copy()).float()
        data270 = torch.from_numpy(data270.copy()).float()

        target0 = torch.from_numpy(np.array(0)).long()
        target90 = torch.from_numpy(np.array(1)).long()
        target180 = torch.from_numpy(np.array(2)).long()
        target270 = torch.from_numpy(np.array(3)).long()

        data_full = torch.stack([data0, data90, data180, data270], dim=0)
        targets_rot = torch.stack([target0, target90, target180, target270], dim=0)

        return data_full, target, targets_rot, index

    def __len__(self):
        return len(self.caltech101)