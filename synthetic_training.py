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

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=64, num_workers=4)

    model = MobilenetV3(num_classes=15069, classifier_activation=nn.Identity)

    # model = resnet50()
    # model.fc = nn.Sequential(nn.Dropout(),
    #                          nn.LazyLinear(4096),
    #                          nn.LazyLinear(8192),
    #                          nn.LazyLinear(15069))

    cfg = get_config()

    flame_layer = FLAME(cfg).to(device)

    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    # aug = nn.Sequential(augmentation.RandomHorizontalFlip(),
    #                     augmentation.RandomAffine(degrees=(-20, 20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15,
    #                                               padding_mode="border")).to(device)

    ckpt_path = "trained_models/mobilenet_synthetic.pt"

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

    for i in range(10):
        train_loss = 0
        model.train()

        pbar = tqdm(train_loader)
        for batch, data in enumerate(pbar):
            inputs, targets = data[0].requires_grad_().to(device), data[1].to(device)
            targets, _ = flame_layer(targets[:, :300], targets[:, 300:400],
                                     targets[:, 400:406], targets[:, 406:409], targets[:, 409:415])
            targets = torch.reshape(targets - torch.mean(targets, dim=1, keepdim=True), (-1, 15069))
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
            targets, _ = flame_layer(targets[:, :300], targets[:, 300:400],
                                     targets[:, 400:406], targets[:, 406:409], targets[:, 409:415])
            targets = torch.reshape(targets - torch.mean(targets, dim=1, keepdim=True), (-1, 15069))

            with torch.no_grad():
                # targets, landmarkst = flame_layer(shape_params=targets, expression_params=exp_params,
                #                                   pose_params=pose_params)
                #
                # outputs, landmarkso = flame_layer(shape_params=model(inputs), expression_params=exp_params,
                #                                   pose_params=pose_params)

                outputs = model(inputs)
                
                loss = loss_fn(outputs, targets)

            # if batch == 0:
            #     for j in range(8):
            #         vertices = outputs[i].detach().cpu().numpy().squeeze()
            #         vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            #         joints = landmarkso[i].detach().cpu().numpy().squeeze()
            #
            #         tri_mesh = trimesh.Trimesh(vertices, faces,
            #                                    vertex_colors=vertex_colors)
            #         mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            #         scene = pyrender.Scene()
            #         scene.add(mesh)
            #         sm = trimesh.creation.uv_sphere(radius=0.005)
            #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            #         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            #         tfs[:, :3, 3] = joints
            #         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            #         scene.add(joints_pcl)
            #         pyrender.Viewer(scene, use_raymond_lighting=True)
            #
            #         vertices = targets[i].detach().cpu().numpy().squeeze()
            #         vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            #         joints = landmarkst[i].detach().cpu().numpy().squeeze()
            #
            #         tri_mesh = trimesh.Trimesh(vertices, faces,
            #                                    vertex_colors=vertex_colors)
            #         mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            #         scene = pyrender.Scene()
            #         scene.add(mesh)
            #         sm = trimesh.creation.uv_sphere(radius=0.005)
            #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            #         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            #         tfs[:, :3, 3] = joints
            #         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            #         scene.add(joints_pcl)
            #         pyrender.Viewer(scene, use_raymond_lighting=True)

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


if __name__ == "__main__":
    main()
