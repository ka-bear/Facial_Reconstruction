from pathlib import Path

import cv2
import numpy as np
import torch
from blendtorch import btt
from torch import nn
from torch.utils import data

from flame.config import get_config
from flame.texture import AlbedoTexture
from flame.flame_pytorch import FLAME


model_file = "../../model/generic_model.pkl"
texture_file = "./model/albedoModel2020_FLAME_albedoPart.npz"


def update_simulations(remotes, vertices, idx):
    n = len(remotes)

    for (remote, vertices_, idx_) in zip(
            remotes,
            torch.chunk(vertices, n),
            torch.chunk(idx, n)
    ):
        remote.send(vertices=vertices_.detach().cpu().numpy(),  idx=idx_.detach().cpu().numpy())


def item_transform(item):
    item["image"] = btt.colors.gamma(item["image"])
    return item


class RandomGenerator(nn.Module):
    def __init__(self, config, texture):
        super(RandomGenerator, self).__init__()
        self.flame_layer = FLAME(config)
        self.texture_model = AlbedoTexture(texture)

        self.register_buffer("shape_params",
                             torch.zeros((config.batch_size // config.images_per_face, config.shape_params),
                                         dtype=torch.float32))
        self.register_buffer("pose_params", torch.zeros((config.batch_size, config.pose_params), dtype=torch.float32))
        self.register_buffer("expression_params",
                             torch.zeros((config.batch_size, config.expression_params), dtype=torch.float32))
        self.register_buffer("texture_params", torch.zeros((config.batch_size // config.images_per_face, 145),
                                                           dtype=torch.float32))
        self.register_buffer("eye_pose", torch.zeros((config.batch_size, 6), dtype=torch.float32))
        self.register_buffer("neck_pose", torch.zeros((config.batch_size, 3), dtype=torch.float32))
        self.register_buffer("transl", torch.concat([torch.zeros(5023, 1), 1.50 * torch.ones(5023, 1),
                                                     torch.zeros(5023, 1)], dim=-1))

        self.config = config

    def forward(self):
        shape_params = (self.shape_params.normal_(0, 1) * 1.1).repeat(self.config.images_per_face, 1)
        pose_params = self.pose_params.normal_(0, 1) * 0.03
        expression_params = self.expression_params.normal_(0, 1) * 0.5
        texture_params = self.texture_params.normal_(0, 1).repeat(self.config.images_per_face, 1)
        eye_pose = self.eye_pose.normal_(0, 1) * 0.1
        neck_pose = self.neck_pose.normal_(0, 1) * 0.2

        vertices, _ = self.flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        vertices = vertices + self.transl

        diff, spec, _ = self.texture_model(texture_params)
        return torch.concat([shape_params, expression_params, pose_params, eye_pose, neck_pose, texture_params],
                            dim=-1), vertices, diff, spec


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up random face generator
    config = get_config()
    rfg = RandomGenerator(config, np.load(texture_file)).to(device)

    SIM_INSTANCES = config.batch_size // config.images_per_face

    launch_args = dict(
        scene=Path(__file__).parent / "faces.blend",
        script=Path(__file__).parent / "faces.blend.py",
        num_instances=SIM_INSTANCES,
        named_sockets=["DATA", "CTRL"],
    )

    with btt.BlenderLauncher(**launch_args) as bl:  # my favorite manga genre
        addr = bl.launch_info.addresses["DATA"]
        sim_ds = btt.RemoteIterableDataset(addr, item_transform=item_transform)
        sim_dl = data.DataLoader(sim_ds, batch_size=config.batch_size, num_workers=1, shuffle=False, timeout=10)

        addr = bl.launch_info.addresses["CTRL"]
        remotes = [btt.DuplexChannel(a) for a in addr]

        sim_iter = iter(sim_dl)

        for i in range(5000):
            shape_params, vertices, diff, spec = rfg()
            idx = torch.arange(i * config.batch_size // config.images_per_face,
                               (i + 1) * config.batch_size // config.images_per_face)
            cv2.imwrite("D:\\python_code\\FacialReconstruction\\tmp\\diff.png", diff[0].detach().cpu().numpy())  # noqa
            cv2.imwrite("D:\\python_code\\FacialReconstruction\\tmp\\spec.png", spec[0].detach().cpu().numpy())  # noqa
            update_simulations(remotes, vertices, idx)

            _ = next(sim_iter)["image"].detach().cpu().numpy()

            np.save(f"D:/python_code/FacialReconstruction/out/shapes/shape_{i}.npy",
                    shape_params.detach().cpu().numpy())


if __name__ == "__main__":
    main()
