import torch
from torch import nn
import numpy as np


class AlbedoTexture(nn.Module):
    def __init__(self, texture_model):
        """
        generates AlbedoMM texture model from loaded file
        in BGR format
        :param texture_model: file loaded from npz
        """
        super(AlbedoTexture, self).__init__()
        n_pc = texture_model['PC'].shape[-1]

        self.register_buffer('MU', torch.tensor(np.reshape(texture_model["MU"], (1, -1)), dtype=torch.float32))
        self.register_buffer('PC', torch.tensor(np.reshape(texture_model["PC"], (-1, n_pc)).T, dtype=torch.float32))
        self.register_buffer('specMU', torch.tensor(np.reshape(texture_model["specMU"], (1, -1)), dtype=torch.float32))
        self.register_buffer('specPC', torch.tensor(np.reshape(texture_model["PC"], (-1, n_pc)).T, dtype=torch.float32))

        self.register_buffer('reshape', torch.tensor([2, 1, 0], dtype=torch.int64))

    def forward(self, tex_params):
        diff = self.MU + torch.matmul(tex_params, self.PC)
        spec = self.specMU + torch.matmul(tex_params, self.specPC)

        diff = torch.reshape(torch.clip(diff, min=0.0, max=1.0), (-1, 512, 512, 3))
        spec = torch.reshape(torch.clip(spec, min=0.0, max=1.0), (-1, 512, 512, 3))

        texture = torch.pow(0.6 * (diff + spec), 1.0/2.2)

        diff = (diff * 255).type(torch.uint8)
        spec = (spec * 255).type(torch.uint8)
        texture = (texture * 255).type(torch.uint8)

        return diff, spec, texture
