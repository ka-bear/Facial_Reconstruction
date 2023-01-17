from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import torch


class CelebADataset(Dataset):
    """
    Dataloader for the CelebA dataset
    """
    def __init__(self, celeba_root, transform=None, train_eval_test=0):
        partition_df = pd.read_csv(celeba_root + "list_eval_partition.csv")
        attr_df = pd.read_csv(celeba_root + "list_attr_celeba.csv")
        attr_df = attr_df[partition_df["partition"] == train_eval_test]
        self.attr = attr_df.iloc[:, 1:42].to_numpy().astype("float32")
        self.image_ids = attr_df["image_id"].to_numpy()
        self.transform = transform
        self.celeba_root = celeba_root

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, idx):
        image = read_image(self.celeba_root + "cropped_and_resized/" + self.image_ids[idx], ImageReadMode.RGB)
        attrs = self.attr[idx]
        image = image.type(torch.FloatTensor)

        if self.transform:
            image = self.transform(image)
        attrs = torch.tensor(attrs)

        return image, attrs
