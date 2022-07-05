import torch
from config import get_config
from flame_pytorch import FLAME

model_file = "model/generic_model.pkl"
texture_file = "model/albedoModel2020_FLAME_albedoPart.npz"


def main():
    config = get_config()
    shape_params = torch.zeros((1, 100), dtype=torch.float32).cuda()
    pose_params = torch.zeros((1, 6), dtype=torch.float32).cuda()
    expression_params = torch.zeros(8, 50, dtype=torch.float32).cuda()

    flame_layer = FLAME(config)
    vertices, landmark = flame_layer(shape_params, expression_params, pose_params)