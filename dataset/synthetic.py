from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import torch


class SyntheticDataset(Dataset):
    def __init__(self, flame_root, transform=None, test=False):
        self.transform = transform
        self.synthetic_root = flame_root
        self.test = test

    def __len__(self):
        if self.test:
            return 8000
        return 32000

    def __getitem__(self, idx):
        if self.test:
            filename = f"render_{4000 + idx // 8}_{idx % 8}.png"
            filename1 = f"shape_{4000 + idx // 8}.npy"

            image = read_image(self.synthetic_root + "/renders//" + filename, ImageReadMode.RGB)
            image = image.type(torch.FloatTensor)
            shape = torch.from_numpy(np.load(self.synthetic_root + "/shapes//" + filename1)[1, :])
        else:
            filename = f"render_{idx // 8}_{idx % 8}.png"
            filename1 = f"shape_{idx // 8}.npy"
            try:
                image = read_image(self.synthetic_root + "/renders//" + filename, ImageReadMode.RGB)
            except Exception:
                print(filename)

            image = image.type(torch.FloatTensor)
            shape = torch.from_numpy(np.load(self.synthetic_root + "/shapes//" + filename1)[1, :])
        if self.transform:
            image = self.transform(image)
        return image, shape
