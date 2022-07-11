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

SIM_INSTANCES = 1


def update_simulations(remotes, vertices, diff, spec):
    n = len(remotes)

    for (remote, vertices_, diff_, spec_) in zip(
            remotes,
            torch.chunk(vertices, n),
            torch.chunk(diff, n),
            torch.chunk(spec, n)
    ):
        remote.send(vertices=vertices_.cpu().numpy(), diff=diff_.cpu().numpy(), spec=spec_.cpu().numpy())
        # todo convert bgr to rgb


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
        self.register_buffer("texture_params", torch.zeros((config.batch_size, 145), dtype=torch.float32))
        self.register_buffer("eye_pose", torch.zeros((config.batch_size, 6), dtype=torch.float32))
        self.register_buffer("neck_pose", torch.zeros((config.batch_size, 3), dtype=torch.float32))

        self.config = config

    def forward(self):
        shape_params = (self.shape_params.normal_(0, 1) * 0.03).repeat(self.config.images_per_face, 1)
        pose_params = self.pose_params.normal_(0, 1) * 0.03
        expression_params = self.expression_params.normal_(0, 1) * 0.5
        texture_params = self.texture_params.normal_(0, 1)
        eye_pose = self.eye_pose.normal_(0, 1) * 0.03
        neck_pose = self.neck_pose.normal_(0, 1)

        vertices, _ = self.flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        diff, spec, _ = self.texture_model(texture_params)

        return shape_params, vertices, diff, spec


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up random face generator
    config = get_config()
    rfg = RandomGenerator(config, np.load(texture_file)).to(device)

    for i in range(5):
        shape_params, vertices, diff, spec = rfg()
        print(rfg.shape_params)

    cv2.imwrite("tmp/diff.png", (diff.cpu().numpy()).astype('uint8').reshape((8, 512, 512, 3))[0])
    cv2.imwrite("tmp/spec.png", (spec.cpu().numpy()).astype('uint8').reshape((8, 512, 512, 3))[0])

    launch_args = dict(
        scene=Path(__file__).parent / "faces.blend",
        script=Path(__file__).parent / "faces.blend.py",
        num_instances=SIM_INSTANCES,
        named_sockets=["DATA", "CTRL"],
    )

    with btt.BlenderLauncher(**launch_args) as bl:  # my favorite manga genre
        addr = bl.launch_info.addresses["DATA"]
        sim_ds = btt.RemoteIterableDataset(addr, item_transform=item_transform)
        sim_dl = data.DataLoader(sim_ds, batch_size=config.batch_size, num_workers=0, shuffle=False)

        addr = bl.launch_info.addresses["CTRL"]
        remotes = [btt.DuplexChannel(a) for a in addr]

        sim_iter = iter(sim_dl)

        for i in range(1000):
            shape_params, vertices, diff, spec = rfg()
            update_simulations(remotes, shape_params, vertices, diff)

            image = next(sim_iter)


    # flame_layer = FLAME(config).to(device)
    #
    # # set up texture model
    # texture_model = AlbedoTexture(np.load(texture_file)).to(device)
    #
    # shape_params = torch.randn((8, config.shape_params), dtype=torch.float32).to(device)
    # pose_params = torch.randn((8, config.pose_params), dtype=torch.float32).to(device) * 0.02
    # expression_params = torch.randn(8, config.expression_params, dtype=torch.float32).to(device) * 0.5
    # texture_params = torch.randn((8, 145), dtype=torch.float32).to(device)
    #
    # vertices, landmarks = flame_layer(shape_params, expression_params, pose_params)
    # faces = flame_layer.faces
    #
    #
    # diff, spec, texture = texture_model(texture_params)

    # for i in range(8):
    #     plt.imshow(np.reshape(diff.detach().cpu().numpy(), (8, 512, 512, 3))[i, :, :, (2, 1, 0)].transpose([1, 2, 0]))
    #     plt.show()

    # vertices = vertices[0].detach().cpu().numpy().squeeze()
    # joints = landmarks[0].detach().cpu().numpy().squeeze()
    #
    # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    #
    # tri_mesh = trimesh.Trimesh(vertices, faces,
    #                            vertex_colors=vertex_colors)
    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # sm = trimesh.creation.uv_sphere(radius=0.005)
    # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    # tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    # tfs[:, :3, 3] = joints
    # joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    # scene.add(joints_pcl)
    # pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    main()
