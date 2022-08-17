import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import trimesh
import os
from torchvision.io import read_image
from torchvision.transforms.functional import resized_crop


class Biwi(Dataset):
    def __init__(self, root, training=True):
        self.root = root
        self.images = sorted(glob(root + "*/*rgb.png"))
        self.images = [image.replace("\\", "/") for image in self.images]
        self.current_subject = ""
        self.training = training
        # cache obj because loading the obj is very slow
        self.objs = dict()
        if not training:
            self.images = self.images[:len(self.images) // 5]
        else:
            self.images = self.images[len(self.images) // 5:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        split = image_path.split("/")

        mat = torch.Tensor(np.loadtxt("_".join(image_path.split('_')[:-1]) + "_pose.txt"))

        current_subject = split[-2]

        vertices = self.objs.get(current_subject)
        # read obj
        if vertices is None:
            with open(os.path.join(*split[:-2], current_subject + ".obj")) as f:
                obj = trimesh.load(f, file_type='obj')
                vertices = torch.Tensor(obj.vertices).unsqueeze(-1) / 1000
                self.objs.update({current_subject: vertices})

        vertices = torch.matmul(mat[:3, ...], vertices)
        projected = vertices + mat[3, ...].unsqueeze(-1) / 1000

        # read calibration (mostly for cropping)
        int_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), max_rows=3))

        ext_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=5, max_rows=3))
        transl = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=10, max_rows=1)) / 1000

        # extrinsic matrix
        projected = torch.matmul(ext_mat, projected) + transl

        # intrinsic matrix
        projected = torch.matmul(int_mat, projected)

        x_min = torch.min(projected[:, 0, ...])
        x_max = torch.max(projected[:, 0, ...])
        y_min = torch.min(projected[:, 1, ...])
        y_max = torch.max(projected[:, 1, ...])

        width = x_max - x_min
        height = y_max - y_min

        if width > height:
            y_min -= (width - height) / 2
            y_max += (width - height) / 2

            # expand crop slightly
            x_min -= width * 0.2
            x_max += width * 0.2
            y_min -= width * 0.2
            y_max += width * 0.2
            size = (width * 1.4).type(torch.int)
        else:
            x_min -= (height - width) / 2
            x_min += (height - width) / 2

            # expand crop slightly
            x_min -= height * 0.2
            x_max += height * 0.2
            y_min -= height * 0.2
            y_max += height * 0.2
            size = (height * 1.4).type(torch.int)

        top = y_min.type(torch.int)
        left = x_min.type(torch.int)

        img = read_image(image_path).float()
        img = resized_crop(img, top, left, size, size, [256, 256])

        return img, torch.flatten(vertices)