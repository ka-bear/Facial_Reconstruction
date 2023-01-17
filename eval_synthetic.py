import os

import torch
import torchvision.transforms as transforms
from kornia import augmentation
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from dataset.synthetic import SyntheticDataset
from flame.config import get_config
from flame.flame_pytorch import FLAME
from model.mobilenet import MobilenetV3
import numpy as np
import trimesh
import pyrender


def main():
    flame_root = "data/synthetic"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
    train_ds = SyntheticDataset(flame_root, transform)
    valid_ds = SyntheticDataset(flame_root, transform, test=True)

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=16, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=16, num_workers=4)

    # model = MobilenetV3(num_classes=15069, classifier_activation=nn.Identity)

    model = resnet50()
    model.fc = nn.Sequential(nn.Dropout(),
                             nn.LazyLinear(4096),
                             nn.LazyLinear(8192),
                             nn.LazyLinear(15069))

    cfg = get_config()

    flame_layer = FLAME(cfg).to(device)

    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    # aug = nn.Sequential(augmentation.RandomHorizontalFlip(),
    #                     augmentation.RandomAffine(degrees=(-20, 20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15,
    #                                               padding_mode="border")).to(device)

    ckpt_path = "trained_models/resnet_synthetic.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr.load_state_dict(ckpt["lr"])
        epoch = ckpt["epoch"]
        metrics = ckpt["metrics"]
        print(f"Checkpoint loaded at epoch {epoch}")
    else:
        epoch = 0
        metrics = {"train": [], "val": []}

    del ckpt

    train_loss = 0
    mean = torch.zeros(size=(15069,), device=device)
    ss = torch.zeros(size=(15069,), device=device)

    optimizer.zero_grad()
    model.eval()

    test_loss = 0
    pbar = tqdm(valid_loader)
    for batch, data in enumerate(pbar):
        inputs, targets = data[0].to(device), data[1].to(device)
        targets, _ = flame_layer(targets[:, :300], targets[:, 300:400],
                                 targets[:, 400:406], targets[:, 406:409], targets[:, 409:415])
        targets = torch.reshape(targets - torch.mean(targets, dim=1, keepdim=True), (-1, 15069))
        mean += torch.mean(targets, dim=0)
        ss += torch.sum(targets * targets, dim=0)

        with torch.no_grad():
            # targets, landmarkst = flame_layer(shape_params=targets, expression_params=exp_params,
            #                                   pose_params=pose_params)
            #
            # outputs, landmarkso = flame_layer(shape_params=model(inputs), expression_params=exp_params,
            #                                   pose_params=pose_params)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

        test_loss += loss.detach()

        pbar.set_description(f"Valid loss: {test_loss / (batch + 1)}")
    metrics["val"].append(test_loss)

    print("RMSE: ", torch.sqrt(test_loss / len(valid_loader)))
    print((mean / len(valid_loader)) ** 2, (ss / len(valid_loader) / 64), test_loss / len(valid_loader))
    print(torch.mean((ss / len(valid_loader) / 64 - (mean / len(valid_loader)) ** 2)))
    print("R2: ", 1 - torch.mean(test_loss / len(valid_loader) / (ss / len(valid_loader) / 64 - (mean / len(valid_loader)) ** 2)))


if __name__ == "__main__":
    main()
