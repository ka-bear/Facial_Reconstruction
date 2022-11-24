import os

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transforms

from dataset.biwi import Biwi
from flame.flame_pytorch import FLAME
import pyrender
import trimesh
#from config import get_config
from flame.config import get_config


def visualise_output():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    biwi_root = "D:\\python_code\\faces_0\\"

    model = resnet50()
    model.fc = nn.Sequential(nn.Dropout(),
                             nn.LazyLinear(4096),
                             nn.LazyLinear(8192),
                             nn.LazyLinear(20754))

    model = model.to(device)

    loss_fn = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    ckpt_path = "modelbiwi.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt["epoch"]
        print(f"Checkpoint loaded at epoch {epoch}")
        del ckpt

    model.eval()

    transform = transforms.Normalize((127.0, 127.0, 127.0), (127.0, 127.0, 127.0))
    valid_ds = Biwi(biwi_root, False)

    valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=16, num_workers=4)

    data = next(iter(valid_loader))

    with torch.no_grad():
        inputs, targets = transform(data[0].to(device)), data[1].to(device)
        vertice = model(inputs)
        vertice = torch.reshape(vertice, (-1, 6918, 3))

    with open(os.path.join(r"D:\python_code\faces_0\01.obj")) as f:
        obj = trimesh.load(f, file_type='obj')
        vertices = obj.vertices
        faces = obj.faces

    # Visualize Landmarks
    targets = torch.reshape(targets, (-1, 6918, 3))
    vertices = vertice[0].detach().cpu().numpy().squeeze()
    print(vertices)
    print(targets[0])
    print(loss_fn(vertice, targets))
    print(torch.std(vertice, dim=0))
    print(torch.std(targets, dim=0))

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    tri_mesh = trimesh.Trimesh(vertices, faces,
                               vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    pyrender.Viewer(scene, use_raymond_lighting=True)

    tri_mesh = trimesh.Trimesh(targets[0].detach().cpu().numpy(), faces,
                               vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    visualise_output()