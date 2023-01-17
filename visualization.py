import os

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transforms

from dataset.biwi import Biwi
import pyrender
import trimesh
import matplotlib.pyplot as plt

from dataset.synthetic import SyntheticDataset
from flame.config import get_config
from flame.flame_pytorch import FLAME
from model.mobilenet import MobilenetV3


def visualise_output():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    biwi_root = "D:\\python_code\\faces_0\\"
    flame_root = "data/synthetic"

    model = resnet50()
    model.fc = nn.Sequential(nn.Dropout(),
                             nn.LazyLinear(4096),
                             nn.LazyLinear(8192),
                             nn.LazyLinear(15069))
    # model = MobilenetV3(num_classes=15069, classifier_activation=nn.Identity)
    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    ckpt_path = "trained_models/resnet_synthetic.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt["epoch"]
        print(f"Checkpoint loaded at epoch {epoch}")
        del ckpt

    model.eval()

    transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
    # valid_ds = Biwi(biwi_root, False)
    valid_ds = SyntheticDataset(flame_root, test=True)

    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=16, num_workers=1)

    data = next(iter(valid_loader))

    with torch.no_grad():
        inputs, targets = transform(data[0].to(device)), data[1].to(device)
        vertices = model(inputs)
        vertices = torch.reshape(vertices, (-1, 5023, 3))

    # with open(os.path.join(r"D:\python_code\faces_0\01.obj")) as f:
    #     obj = trimesh.load(f, file_type='obj')
    #     faces = obj.faces
    cfg = get_config()

    flame_layer = FLAME(cfg).to(device)
    faces = flame_layer.faces

    for i in range(16):
        plt.imsave(f"face{i}.png", np.clip(inputs[i].permute(1, 2, 0).detach().cpu().numpy() + 0.5, a_min=0, a_max=1))
        plt.show()

    # Visualize Landmarks
    # targets = torch.reshape(targets, (-1, 5023, 3))
    vertices = vertices.detach().cpu().numpy().squeeze()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    for i, vertex in enumerate(vertices):
        mesh = trimesh.Trimesh(vertex, faces,
                               vertex_colors=vertex_colors)
        mesh.export(f"face{i}.obj", file_type="obj")

    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # sm = trimesh.creation.uv_sphere(radius=0.005)
    # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    #
    # tri_mesh = trimesh.Trimesh(targets[0].detach().cpu().numpy(), faces,
    #                            vertex_colors=vertex_colors)
    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # sm = trimesh.creation.uv_sphere(radius=0.005)
    # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    # pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    visualise_output()
