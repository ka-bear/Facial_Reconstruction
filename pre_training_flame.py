import os

import torch
import torchvision.transforms as transforms
from kornia import augmentation
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import numpy as np
from model.mobilenet import MobilenetV3

flame_root = "/content/monopoly"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
train_ds = FlameDataset(flame_root, transform)
valid_ds = FlameDataset(flame_root, transform)

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=False, batch_size=64, num_workers=4)

model = MobilenetV3(num_classes=300, classifier_activation=nn.Identity).to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

aug = nn.Sequential(augmentation.RandomHorizontalFlip(),
                    augmentation.RandomAffine(degrees=(-20, 20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15,
                                              padding_mode="border")).to(device)

ckpt_path = "model.pt"

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

for i in range(10):
    train_loss = 0
    model.train()

    pbar = tqdm(train_loader)
    for batch, data in enumerate(pbar):
        inputs, targets = aug(data[0].requires_grad_().to(device)), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach()

        pbar.set_description(f"Batch: {batch} Training loss: {train_loss / (batch + 1)}")

    metrics["train"].append(train_loss)

    optimizer.zero_grad()
    model.eval()

    test_loss = 0
    pbar = tqdm(valid_loader)
    for batch, data in enumerate(pbar):
        inputs, targets = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        test_loss += loss.detach()

        pbar.set_description(f"Valid loss: {test_loss / (batch + 1)}")
    metrics["val"].append(test_loss)

    epoch += 1
    lr.step(test_loss)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr": lr.state_dict(),
        "metrics": metrics,
        "epoch": epoch
    }, ckpt_path)

    print(f"Epoch {epoch} done")
