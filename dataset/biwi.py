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
            mat_ = torch.concat([mat[:3, :], mat[3].view((3, 1)) / 1000], dim=1)
            mat_ = torch.concat([mat_, torch.Tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            self.current_subject = split[-2]
            # read obj
            with open(os.path.join(*split[:-2], split[-2] + ".obj")) as f:
                obj = trimesh.load(f, file_type='obj')

            self.vertices = torch.Tensor(obj.vertices) / 1000

            # read calibration (mostly for cropping)
            int_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), max_rows=3))
            int_mat = torch.concat([int_mat, torch.zeros(size=(3,1))], dim=1)

            # print(mat_)

            ext_mat = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=5, max_rows=3))
            transl = torch.Tensor(np.loadtxt(os.path.join(*split[:-1], "rgb.cal"), skiprows=10, max_rows=1)) / 1000
            ext_mat = torch.concat([ext_mat, transl.view(3, 1)], dim=1)
            # print(ext_mat.shape)
            ext_mat = torch.concat([ext_mat, torch.Tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            # print(ext_mat)


            # translation and rotation of head
            # print(self.vertices.shape)
            vertices = torch.matmul(mat_, torch.concat([self.vertices, torch.ones((self.vertices.shape[0], 1))], dim=1).unsqueeze(-1))
            # print(vertices)
            # print(vertices.shape)
            # print(ext_mat.shape)

            # extrinsic matrix
            projected = torch.matmul(ext_mat, vertices)
            # print(vertices)
            # projected = torch.Tensor([1.0, 1.0, -1.0, 1.0]).view(1, 4, 1) * projected
            # print(torch.min(projected), torch.max(projected))
            # print(projected.shape, projected.squeeze())
            # projected = projected + transl.view((3, 1))

            # intrinsic matrix
            projected = torch.matmul(int_mat, projected)
            # print(projected)

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
                self.size = (width * 1.4).type(torch.int)
            else:
                x_min -= (height - width) / 2
                x_min += (height - width) / 2

                # expand crop slightly
                x_min -= height * 0.2
                x_max += height * 0.2
                y_min -= height * 0.2
                y_max += height * 0.2
                self.size = (height * 1.4).type(torch.int)

            self.top = y_min.type(torch.int)
            self.left = x_min.type(torch.int)

        img = read_image(image_path).float()
        img = resized_crop(img, self.top, self.left, self.size, self.size, [256, 256])

        vertices = torch.matmul(mat[0:3, :], self.vertices.unsqueeze(-1))
        vertices = vertices + mat[3, :].unsqueeze(0)

        return img, vertices