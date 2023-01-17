import os
from functools import partial

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
    biwi_root = "D:\\python_code\\faces_0\\"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
    valid_ds = Biwi(biwi_root, False)

    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=16, num_workers=4)

    # model = MobilenetV3(num_classes=20754, classifier_activation=nn.Identity)

    model = resnet50()
    model.fc = nn.Sequential(nn.Dropout(),
                             nn.LazyLinear(4096),
                             nn.LazyLinear(8192),
                             nn.LazyLinear(20754))

    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    ckpt_path = "trained_models/resnet_biwi.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt["epoch"]
        print(f"Checkpoint loaded at epoch {epoch}")
        del ckpt

    model.eval()

    test_loss = torch.Tensor((0,)).to(device)
    mean = torch.zeros(size=(20754,)).to(device)

    pbar = tqdm(valid_loader)
    for batch, data in enumerate(pbar):
        targets = data[1].to(device)
        with torch.no_grad():
            mean += torch.mean(targets, 0).detach()

    mean = mean / len(valid_loader)

    sse = torch.Tensor((0,)).to(device)
    sst = torch.Tensor((0,)).to(device)
    mae = torch.Tensor((0,)).to(device)

    pbar = tqdm(valid_loader)
    for batch, data in enumerate(pbar):
        inputs, targets = transform(data[0].to(device)), data[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            sse += loss.detach()
            sst += torch.mean(torch.pow(targets - mean, 2))
            mae += l1_loss(outputs, targets).detach()

        pbar.set_description(f"Valid loss: {test_loss / (batch + 1)}")
    print(1 - sse/sst)
    print(torch.sqrt(sse / len(valid_loader)))
    print(mae / len(valid_loader))


if __name__ == "__main__":
    main()
