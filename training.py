import os

import torch
import torchvision.transforms as transforms
from kornia import augmentation
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from dataset.biwi import Biwi
from model.mobilenet import MobilenetV3


def main():
    torch.cuda.empty_cache()

    biwi_root = r"D:\python_code\faces_0/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
    train_ds = Biwi(biwi_root, True)
    valid_ds = Biwi(biwi_root, False)

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=64, num_workers=4)

    model = MobilenetV3(num_classes=20754, classifier_activation=nn.Identity)
    #
    # model = resnet50()
    # model.fc = nn.Sequential(nn.Dropout(),
    #                          nn.LazyLinear(4096),
    #                          nn.LazyLinear(8192),
    #                          nn.LazyLinear(20754))

    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    aug = nn.Sequential(augmentation.RandomHorizontalFlip(),
                        augmentation.RandomAffine(degrees=(-20, 20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15,
                                                  padding_mode="border")).to(device)

    ckpt_path = "mobilenet_biwi.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr.load_state_dict(ckpt["lr"])
        epoch = ckpt["epoch"]
        metrics = ckpt["metrics"]
        print(f"Checkpoint loaded at epoch {epoch}")
        del ckpt
    else:
        epoch = 0
        metrics = {"train": [], "val": []}

    for i in range(50):
        train_loss = 0
        model.train()

        pbar = tqdm(train_loader)
        for batch, data in enumerate(pbar):
            inputs, targets = aug(transform(data[0].requires_grad_().to(device))), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu()

            pbar.set_description(f"Batch: {batch} Training loss: {train_loss / (batch + 1)}")

        metrics["train"].append(train_loss)

        optimizer.zero_grad()
        model.eval()

        test_loss = 0
        pbar = tqdm(valid_loader)
        for batch, data in enumerate(pbar):
            inputs, targets = transform(data[0].to(device)), data[1].to(device)

            with torch.no_grad():

                outputs = model(inputs)

                loss = loss_fn(outputs, targets)

            test_loss += loss.detach().cpu()

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


if __name__ == "__main__":
    main()
