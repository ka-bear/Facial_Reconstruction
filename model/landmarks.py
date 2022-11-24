from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from torch import nn


def main():
    pose_model = init_pose_model("mmpose_config.py", "../models/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth")


if __name__=="__main__":
    main()
