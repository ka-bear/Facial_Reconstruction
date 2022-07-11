import torch
from torch import nn
import numpy as np


class AlbedoTexture(nn.Module):
    def __init__(self, texture_model):
        '''
        generates AlbedoMM texture model from loaded file
        :param texture_model: file loaded from npz
        '''
        super(AlbedoTexture, self).__init__()
        num_tex_pc = texture_model['PC'].shape[-1]

        self.register_buffer('MU', torch.tensor(np.reshape(texture_model["MU"], (1, -1)), dtype=torch.float32))
        self.register_buffer('PC', torch.tensor(np.reshape(texture_model["PC"], (-1, num_tex_pc)).T, dtype=torch.float32))
        self.register_buffer('specMU', torch.tensor(np.reshape(texture_model["specMU"], (1, -1)), dtype=torch.float32))
        self.register_buffer('specPC', torch.tensor(np.reshape(texture_model["PC"], (-1, num_tex_pc)).T, dtype=torch.float32))

    def forward(self, tex_params):
        diff = self.MU + torch.matmul(tex_params, self.PC)
        spec = self.specMU + torch.matmul(tex_params, self.specPC)

        diff = torch.reshape(255 * torch.clip(diff, min=0.0, max=1.0), (-1, 512, 512, 3))
        spec = torch.reshape(255 * torch.clip(spec, min=0.0, max=1.0), (-1, 512, 512, 3))

        texture = torch.pow(0.6 * (diff + spec), 1.0/2.2)

        return diff, spec, texture
