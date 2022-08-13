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
    cfg = get_config()
    flame_layer = FLAME(cfg)
    params = torch.tensor(np.load("E:\\pycharmgoesbrr\\Facial_Reconstruction\\data\\synthetic\\shapes\\shape_1.npy"))

    shape_params = params[:, :300]
    # expression_params = params[:, 300:400]
    # pose_params = params[:, 400:406]
    # eye_pose = params[:, 406:412]
    # neck_pose = params[:, 412:415]
    expression_params = torch.zeros((8,100))
    pose_params = torch.zeros((8,6))
    eye_pose = torch.zeros((8, 6))
    neck_pose = torch.zeros((8,3))

    faces = flame_layer.faces

    vertcies, landmarks = flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    for j in range(8):
        vertices = vertcies[j].detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        joints = landmarks[j].detach().cpu().numpy().squeeze()

        tri_mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    main()
