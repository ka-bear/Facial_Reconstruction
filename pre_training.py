import os

from dataset.celeb_a import CelebADataset
from model.losses.focal_loss import FocalLoss
from model.mobilenet import MobilenetV3

from tqdm import tqdm

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

celeba_root = "E:\\pycharmgoesbrr\\Project-Vitello-Tonnato\\data\\celeba\\img_celeba\\"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                                            shear=15,
                                                            interpolation=transforms.InterpolationMode.BILINEAR)])
    train_ds = CelebADataset(celeba_root, transform, 0)
    valid_ds = CelebADataset(celeba_root, None, 1)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=False, batch_size=32, num_workers=4)

    model = MobilenetV3(num_classes=40, classifier_activation=nn.Sigmoid).to(device)
    loss_fn = FocalLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

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

    model.train()

    for i in range(1):
        train_loss = 0
        model.train()
        for batch, data in enumerate(pbar := tqdm(train_loader)):
            inputs, targets = data[0].requires_grad_().to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()

            pbar.set_description(f"Batch: {batch} Training loss: {train_loss / (batch + 1)}")

            if batch == 100:
                break
        metrics["train"].append(train_loss)

        optimizer.zero_grad()
        model.eval()

        test_loss = 0
        for batch, data in enumerate(pbar := tqdm(valid_loader)):
            inputs, targets = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            test_loss += loss.detach()

            pbar.set_description(f"Valid loss: {test_loss / (batch + 1)}")
        metrics["val"].append(test_loss)

        epoch += 1
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": lr.state_dict(),
            "metrics": metrics,
            "epoch": epoch
        }, ckpt_path)
        lr.step(test_loss)


if __name__ == "__main__":
    main()
