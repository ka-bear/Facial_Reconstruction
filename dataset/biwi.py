import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import trimesh
import os
from torchvision.io import read_image
from torchvision.transforms.functional import resized_crop


class Biwi(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = sorted(glob(root + "*/*rgb.png"))
        self.images = [image.replace("\\", "/") for image in self.images]
        self.current_subject = ""

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        split = image_path.split("/")

        mat = torch.Tensor(np.loadtxt("_".join(image_path.split('_')[:-1]) + "_pose.txt"))

        # if new subject
        if self.current_subject != split[-2]:
            self.current_subject = split[-2]
            # read obj
            obj = trimesh.load(os.path.join(*split[:-2], split[-2] + ".obj"))
            self.vertices = torch.Tensor(obj.vertices)

            # read calibration (mostly for cropping)
            int_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), max_rows=3)) / 640
            ext_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=5, max_rows=3))
            transl = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=10, max_rows=1))

            # translation and rotation of head
            vertices = torch.matmul(mat[:3, :], self.vertices.unsqueeze(-1))
            vertices = vertices + mat[3, :].unsqueeze(0)

            # extrinsic matrix
            projected = torch.matmul(ext_mat, self.vertices.unsqueeze(-1))
            projected = projected + transl.view((3, 1))

            # intrinsic matrix
            projected = torch.matmul(int_mat, projected)

            x_min = torch.min(projected[:, 0, ...]) + 320
            x_max = torch.max(projected[:, 0, ...]) + 320
            y_min = torch.min(projected[:, 1, ...]) + 240
            y_max = torch.max(projected[:, 1, ...]) + 240

            width = x_max - x_min
            height = y_max - y_min

            if width > height:
                y_min -= (width - height) / 2
                y_max += (width - height) / 2

                # expand crop slightly
                x_min -= width * 0.1
                x_max += width * 0.1
                y_min -= width * 0.1
                y_max += width * 0.1
                self.size = width.type(torch.int)
            else:
                x_min -= (height - width) / 2
                x_min += (height - width) / 2

                # expand crop slightly
                x_min -= height * 0.1
                x_max += height * 0.1
                y_min -= height * 0.1
                y_max += height * 0.1
                self.size = height.type(torch.int)

            self.top = y_min.type(torch.int)
            self.left = x_min.type(torch.int)

        img = read_image(image_path).float()
        img = resized_crop(img, self.top, self.left, self.size, self.size, [256, 256])

        vertices = torch.matmul(mat[0:3, :], self.vertices.unsqueeze(-1))
        vertices = vertices + mat[3, :].unsqueeze(0)

        return img, vertices
